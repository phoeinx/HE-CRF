package main

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"regexp"
	"strconv"
	"time"

	"github.com/ChristianMct/helium"
	"github.com/ChristianMct/helium/circuits"
	"github.com/ChristianMct/helium/objectstore"
	"github.com/ChristianMct/helium/protocols"
	"github.com/ChristianMct/helium/services/compute"
	"github.com/ChristianMct/helium/services/setup"
	"github.com/ChristianMct/helium/sessions"
	"github.com/tuneinsight/lattigo/v5/core/rlwe"
	"github.com/tuneinsight/lattigo/v5/he"
	"github.com/tuneinsight/lattigo/v5/mhe"
	"github.com/tuneinsight/lattigo/v5/schemes/bgv"

	"github.com/ChristianMct/helium/node"
)

// cloudAddress is the local address that the cloud node listens on.
const cloudAddress = ":40000"

// defines the command-line flags
var (
	nodeId       = flag.String("node_id", "", "the id of the node")
	nParty       = flag.Int("n_party", -1, "the number of parties")
	cloudAddr    = flag.String("cloud_address", "", "the address of the helper node")
	argThreshold = flag.Int("threshold", -1, "the threshold")
	expRounds    = flag.Int("expRounds", 1, "number of circuit evaluation rounds to perform")
	nEstimators  = flag.Int("nEstimators", 100, "number of estimators to use for Completely Random Forest")
	treeDepth   = flag.Int("treeDepth", 3, "depth of the trees")
	nonParticipationProb = flag.Float64("nonParticipationProb", -1, "probability of a node not participating in the evaluation of a record (default: 0.0, i.e., all nodes participate)")
)

func main() {

	flag.Parse()

	if *nParty < 2 {
		panic("n_party argument should be provided and > 2")
	}

	if len(*nodeId) == 0 {
		panic("node_id argument should be provided")
	}

	if len(*cloudAddr) == 0 {
		panic("cloud_address argument must be provided for session nodes")
	}

	if *nEstimators < 1 {
		panic("nEstimators argument should be provided and > 1")
	}
	if *treeDepth < 1 {
		panic("treeDepth argument should be provided and > 1")
	}

	if *nonParticipationProb < 0.0 || *nonParticipationProb > 1.0 {
		panic("nonParticipationProb argument must be between 0.0 and 1.0")
	}

	nEstimators := *nEstimators
	treeDepth := *treeDepth
	nonParticipationProb := *nonParticipationProb

	// sets the threshold to the number of parties if not provided
	var threshold int
	switch {
	case *argThreshold == -1:
		threshold = *nParty
	case *argThreshold > 0 && *argThreshold <= *nParty:
		threshold = *argThreshold
	default:
		flag.Usage()
		panic("threshold argument must be between 1 and N")
	}

	nid := sessions.NodeID(*nodeId)

	// generates a test node list from the command-line arguments
	nids, nl, shamirPks, nodeMapping := genNodeLists(*nParty, *cloudAddr)

	// generates a nodeConfig for the node running this program
	nodeConfig := genConfigForNode(nid, nids, threshold, shamirPks)

	dataFolderPath, attributeDomainsPath, modelPath := readExperimentConfig()

	// retrieves the session parameters from the node config
	params, err := bgv.NewParametersFromLiteral(nodeConfig.SessionParameters[0].FHEParameters.(bgv.ParametersLiteral))
	if err != nil {
		panic(err)
	}
	fmt.Println("Running with", *nParty, "parties")
	fmt.Println("and threshold", threshold)
	// the maximum number of slots in a plaintext and length of the vectors
	v := params.MaxSlots()

	fmt.Println("Max slots", v)


	// generates the Helium application (see helium/node/app.go).
	app := getApp(params)

	// creates a context for the session
	ctx := sessions.NewBackgroundContext(nodeConfig.SessionParameters[0].ID)

	// creates a random number generator for reproducibility
	const seed int64 = 42
	var randGen = rand.New(rand.NewSource(seed))

	// runs Helium as a server or client
	var timeSetup, timeCompute time.Duration
	var stats map[string]interface{}
	start := time.Now()
	if nodeConfig.ID == "cloud" {
		
		// Runs the Helium server. The method returns when the setup phase has completed.
		// It returns a channel to send circuit descriptors (evaluation requests) and a channel to
		// receive the evaluation outputs.
		hsv, cdescs, outs, err := helium.RunHeliumServer(ctx, nodeConfig, nl, app, compute.NoInput)
		
		if err != nil {
			log.Fatalf("error running helium server: %v", err)
		}

		timeSetup = time.Since(start)
		fmt.Println("Time Setup", timeSetup)
		
		start = time.Now()
		attributeDomains := ReadAttributeDomains(attributeDomainsPath)
		trees := CreateTreeStructures(attributeDomains, treeDepth, nEstimators, randGen)

		// Calculate the number of trees that can be calculated in one circuit
		numLeaves := (1 << treeDepth) * 2 * nEstimators 
		numCircuits := (numLeaves + v - 1) / v
		treesPerCircuit := (nEstimators + numCircuits - 1) / numCircuits

		// Split tree structures into chunks per circuit and transform them to JSON Strings to pass them as arguments
		treeStructureMap := make(map[string]string)
		for nSig := 0; nSig < numCircuits; nSig++ {
			startSliceIndex := nSig * treesPerCircuit
			endSliceIndex := min(startSliceIndex + treesPerCircuit, len(trees))
			circuitTrees := trees[startSliceIndex : endSliceIndex]

			treesString, _ := json.Marshal(circuitTrees)
			treeStructureMap[fmt.Sprintf("vecadd-%d", nSig)] = string(treesString)
		}

		go func() {
			for nSig := 0; nSig < numCircuits; nSig++ {
				sigID := fmt.Sprintf("vecadd-%d", nSig)
				cdescs <- circuits.Descriptor{
					Signature:   circuits.Signature{Name: "vecadd-dec", Args: map[string]string{"treeStructures": treeStructureMap[sigID], "n_party": strconv.Itoa(*nParty)}},
					CircuitID:   sessions.CircuitID(sigID),
					NodeMapping: nodeMapping,
					Evaluator:   "cloud",
				}
			}
			close(cdescs)
		}()

		for out := range outs {
			encoder := bgv.NewEncoder(params)
			pt := &rlwe.Plaintext{Element: out.Ciphertext.Element, Value: out.Ciphertext.Value[0]}
			pt.IsBatched = true
			aggregatedLeafCounts := make([]uint64, params.MaxSlots())

			err := encoder.Decode(pt, aggregatedLeafCounts)
			if err != nil {
				log.Fatalf("%s | [main] error decoding output: %v\n", nodeConfig.ID, err)
			}
			if err != nil {
				log.Fatalf("%s | [main] error decoding output: %v\n", nodeConfig.ID, err)
			}

			updateModel(out.CircuitID, trees, treesPerCircuit, aggregatedLeafCounts)
			
		}

		writeOutModel(modelPath, trees, nodeConfig)

		hsv.GracefulStop() // waits for the last client to disconnect
		timeCompute = time.Since(start)
		stats = map[string]interface{}{
			"Time": map[string]interface{}{
				"Setup":   timeSetup,
				"Compute": timeCompute,
			},
			"Net": hsv.GetStats(),
		}
	} else {

		records := readNodeData(dataFolderPath, nid)

		encoder := bgv.NewEncoder(params)
		var ip compute.InputProvider = getInputProvider(params, encoder, v, nid, records, nonParticipationProb)
		secrets := loadSecrets(nodeConfig.SessionParameters[0], nid)

		// runs the Helium client. The method returns a channel to receive the evaluation outputs
		// for which the node is the receiver.
		hc, outs, err := helium.RunHeliumClient(ctx, nodeConfig, nl, secrets, app, ip)
		if err != nil {
			log.Fatalf("error running helium client: %v", err)
		}

		out, hasOut := <-outs
		if hasOut {
			log.Fatalf("Node %s received output: %v", nodeConfig.ID, out)
		}

		if err := hc.Close(); err != nil {
			log.Fatalf("error closing helium client: %v", err)
		}

		stats = map[string]interface{}{
			"net": hc.GetStats(),
		}
	}

	//outputs the stats as JSON on stdout
	statsJson, err := json.Marshal(stats)
	if err != nil {
		log.Fatalf("error marshalling stats: %v", err)
	}
	fmt.Println("STATS", string(statsJson))
}

func readNodeData(dataFolderPath string, nid sessions.NodeID) ([][]float64) {
	filename := fmt.Sprintf("%s/%s.csv", dataFolderPath, nid)
	file, err := os.Open(filename)
	if err != nil {
		log.Fatalf("Error opening file:", err)
	}
	defer file.Close()
	reader := csv.NewReader(file)

	_, err = reader.Read()
	if err != nil {
		log.Fatalf("Error reading file:", err)
	}
	records, err := reader.ReadAll()
	if err != nil {
		log.Fatalf("Error reading f    ile:", err)
	}

	var floatRecords [][]float64
	for i, row := range records {
		var floatRow []float64
		for j, field := range row {
			val, err := strconv.ParseFloat(field, 64)
			if err != nil {
				log.Fatalf("failed to parse float at row %d, col %d: %w", i, j, err)
			}
			floatRow = append(floatRow, val)
		}
		floatRecords = append(floatRecords, floatRow)
	}
	return floatRecords
}

func writeOutModel(modelPath string, model []PerfectBinaryTree, nodeConfig node.Config) bool {
	filename := modelPath
	file, err := os.Create(filename)
	if err != nil {
		log.Fatalf("Error creating file:", err)
		return true
	}
	defer file.Close()
	jsonEncoder := json.NewEncoder(file)
	jsonEncoder.SetIndent("", "  ")
	err = jsonEncoder.Encode(model)
	if err != nil {
		log.Fatalf("Error encoding JSON:", err)
		return true
	}
	fmt.Printf("Node %s | [main] wrote model to %s\n", nodeConfig.ID, filename)
	return false
}

func updateModel(sigID sessions.CircuitID, trees []PerfectBinaryTree, treesPerCircuit int, aggregatedLeafCounts []uint64) {
	id, err := extractID(string(sigID))
	if err != nil {
		log.Fatalf("error extracting ID from string: %v", err)
	}

	treeSlice := trees[id*treesPerCircuit : (id+1)*treesPerCircuit]

	for treeIndex, tree := range treeSlice {
		firstLeafIndex := (1 << tree.Height) - 1
		leafCount := 1 << (tree.Height + 1)
		for leafIndex := firstLeafIndex; leafIndex < len(tree.Nodes); leafIndex++ {
			countIndex := treeIndex*leafCount + (leafIndex-firstLeafIndex)*2
			totalLeafCount := aggregatedLeafCounts[countIndex] + aggregatedLeafCounts[countIndex+1]
			if totalLeafCount == 0 {
				tree.Nodes[leafIndex].IsEmpty = true
				continue
			}
			tree.Nodes[leafIndex].Prediction[0] = float64(aggregatedLeafCounts[countIndex]) / float64(totalLeafCount)
			tree.Nodes[leafIndex].Prediction[1] = float64(aggregatedLeafCounts[countIndex+1]) / float64(totalLeafCount)

		}
	}

}

func readExperimentConfig() (string, string, string) {
	experimentConfigFile := "/helium/data/experiments/experiment_config.json"
	file, err := os.Open(experimentConfigFile)
	if err != nil {
		log.Fatalf("Error opening file: %v", err)
	}
	defer file.Close()

	decoder := json.NewDecoder(file)
	var experimentConfig map[string]interface{}
	err = decoder.Decode(&experimentConfig)
	if err != nil {
		log.Fatalf("Error decoding JSON: %v", err)
	}
	dataFolderPath := experimentConfig["data_folder_path"].(string)
	attributeDomainsPath := experimentConfig["attribute_domains_path"].(string)
	modelPath := experimentConfig["model_path"].(string)
	return dataFolderPath, attributeDomainsPath, modelPath
}

// genNodeLists generates a test list of node informations from the experiments parameters.
// In a real scenarios, the node informations would be provided by the user application.
func genNodeLists(nParty int, cloudAddr string) (nids []sessions.NodeID, nl node.List, shamirPks map[sessions.NodeID]mhe.ShamirPublicPoint, nodeMapping map[string]sessions.NodeID) {
	nids = make([]sessions.NodeID, nParty)
	nl = make(node.List, nParty)
	shamirPks = make(map[sessions.NodeID]mhe.ShamirPublicPoint, nParty)
	nodeMapping = make(map[string]sessions.NodeID, nParty+2)
	nodeMapping["cloud"] = "cloud"
	for i := range nids {
		nids[i] = sessions.NodeID(fmt.Sprintf("node-%d", i))
		nl[i].NodeID = nids[i]
		shamirPks[nids[i]] = mhe.ShamirPublicPoint(i + 1)
		nodeMapping[string(nids[i])] = nids[i]
	}
	nl = append(nl, struct {
		sessions.NodeID
		node.Address
	}{NodeID: "cloud", Address: node.Address(cloudAddr)})
	return
}

// genConfigForNode generates a node.Config for the node with the provided node ID. It also simulates the loading of the secret-key for the node.
// In a real scenario, the secret-key would be loaded from a secure storage.
func genConfigForNode(nid sessions.NodeID, nids []sessions.NodeID, threshold int, shamirPks map[sessions.NodeID]mhe.ShamirPublicPoint) (nc node.Config) {
	sessParams := sessions.Parameters{
		ID:            "test-session",
		Nodes:         nids,
		//FHEParameters: bgv.ParametersLiteral{PlaintextModulus: 65537, LogN: 14, LogQ: []int{56, 55, 55, 54, 54, 54}, LogP: []int{55, 55}},
		//Gives panic: runtime error: index out of range [4096] with length 4096, works on helium example. TODO: Find other parameters.
		FHEParameters: bgv.ParametersLiteral{PlaintextModulus: 79873, LogN: 12, LogQ: []int{45, 45}, LogP: []int{19}},
		Threshold:     threshold,
		PublicSeed:    []byte{'c', 'r', 's'},
		ShamirPks:     shamirPks,
	}

	nc = node.Config{
		ID:                nid,
		HelperID:          "cloud",
		SessionParameters: []sessions.Parameters{sessParams},
		ObjectStoreConfig: objectstore.Config{BackendName: "mem"},
		TLSConfig:         node.TLSConfig{InsecureChannels: true},
		SetupConfig: setup.ServiceConfig{
			Protocols: protocols.ExecutorConfig{MaxProtoPerNode: 64, MaxParticipation: 64, MaxAggregation: 64},
		},
		ComputeConfig: compute.ServiceConfig{
			MaxCircuitEvaluation: 16,
			Protocols:            protocols.ExecutorConfig{MaxProtoPerNode: 64, MaxParticipation: 64, MaxAggregation: 64},
		},
	}

	if nid == "cloud" {
		//nc.Address = node.Address(cloudAddress)
		nc.SetupConfig.Protocols.MaxAggregation = 64
		nc.ComputeConfig.Protocols.MaxAggregation = 64
	}
	return
}

// getApp generates the Helium application for the test.
// The application specifies the setup phase and declares the circuits that can be executed by the nodes.
func getApp(params bgv.Parameters) node.App {
	return node.App{
		SetupDescription: &setup.Description{
			Cpk: true,
			Rlk: false,
			Gks: []uint64{},
		},
		Circuits: map[circuits.Name]circuits.Circuit{
			"vecadd-dec": vecAddDec,
		},
	}
}

// getInputProvider generates an input provider function for the node. The input provider function
// is registered to with the Helium node and is called by Helium to provide the input for the circuit evaluation.
func getInputProvider(params bgv.Parameters, encoder *bgv.Encoder, m int, nodeID sessions.NodeID, records [][]float64, nonParticipationProb float64) compute.InputProvider {
	return func(ctx context.Context, sess sessions.Session, cd circuits.Descriptor) (chan circuits.Input, error) {

		treesString := cd.Signature.Args["treeStructures"]
		//Unmarshal array of PerfectBinaryTrees
		treeStructures := make([]PerfectBinaryTree, 0)
		err := json.Unmarshal([]byte(treesString), &treeStructures)
		if err != nil {
			return nil, fmt.Errorf("error unmarshalling tree structure: %v", err)
		}

		// filter records
		leafVector := CalculateLeafVector(treeStructures, records, nonParticipationProb)

		encoder := encoder.ShallowCopy()
		pt := bgv.NewPlaintext(params, params.MaxLevelQ())
		err = encoder.Encode(leafVector, pt)

		if err != nil {
			return nil, err
		}

		inchan := make(chan circuits.Input, 1)
		inchan <- circuits.Input{OperandLabel: circuits.OperandLabel(fmt.Sprintf("//%s/%s/vec", nodeID, cd.CircuitID)), OperandValue: pt}
		close(inchan)
		return inchan, nil

	}
}


func vecAddDec(rt circuits.Runtime) error {

	nodeCountParam := rt.Circuit().Signature.Args["n_party"]
	//converts the string to an int
	nodeCount, err := strconv.Atoi(nodeCountParam)
	if err != nil {
		log.Fatalf("error converting nodeCount to int: %v", err)
	}
	inputs := make(map[int]*circuits.FutureOperand)

	
	for i := 0; i < nodeCount; i++ {
		inputs[i] = rt.Input(circuits.OperandLabel(fmt.Sprintf("//node-%d/vec", i)))
	}

	// computes the addition of all input vectors
	opRes := rt.NewOperand("//cloud/res-0")
	if err := rt.EvalLocal(false, nil, func(eval he.Evaluator) error {
		var sum *rlwe.Ciphertext
		var err error
		sum = inputs[0].Get().Ciphertext
		for i := 1; i < nodeCount; i++ {
			if sum, err = eval.AddNew(sum, inputs[i].Get().Ciphertext); err != nil {
				return err
			}
		}
		opRes.Ciphertext = sum
		return err

	}); err != nil {
		return err
	}

	// decrypts the result with result receiver id "cloud". The node id can be a place-holder and the actual id is provided
	// when querying for a circuit's execution.
	return rt.DEC(*opRes, "cloud", map[string]string{
		"smudging": "40.0", // use 40 bits of smudging.
	})
}

// simulates loading the secrets. In a real application, the secrets would be loaded from a secure storage.
func loadSecrets(params sessions.Parameters, nid sessions.NodeID) node.SecretProvider {

	var sp node.SecretProvider = func(sid sessions.ID, nid sessions.NodeID) (*sessions.Secrets, error) {

		if sid != params.ID {
			return nil, fmt.Errorf("no secret for session %s", sid)
		}

		ss, err := sessions.GenTestSecretKeys(params)
		if err != nil {
			return nil, err
		}

		secrets, ok := ss[nid]
		if !ok {
			return nil, fmt.Errorf("node %s not in session", nid)
		}

		return secrets, nil
	}

	return sp
}

type Node struct {
	AttributeIndex int
	Threshold      float64
	IsLeaf         bool
	Prediction	   [2]float64
	IsEmpty	   bool
}

type PerfectBinaryTree struct {
	Nodes  []Node
	Height int
}

func (t PerfectBinaryTree) String() string {
	return fmt.Sprintf("PBT: %d Nodes, %d Height}", len(t.Nodes), t.Height)
}

type AttributeDomain struct {
	Min float64
	Max float64
}

type AttributeDomains map[int]AttributeDomain

func CreateTreeStructures(domains AttributeDomains, height int, count int, randGen *rand.Rand) []PerfectBinaryTree {
	trees := make([]PerfectBinaryTree, 0)

	for i := 0; i < count; i++ {
		tree := CreateTreeStructure(domains, height, randGen)
		trees = append(trees, tree)
	}

	return trees
}

func CreateTreeStructure(domains AttributeDomains, height int, randGen *rand.Rand) PerfectBinaryTree {
	numSplits := int(math.Pow(2, float64(height))) - 1
	nodes := make([]Node, numSplits*2+1)

	getChildIndices := func(index int) (left int, right int) {
		return 2 * index + 1, 2 * index + 2
	}

	copyDomains := func(domains AttributeDomains) map[int]AttributeDomain {
		domainsCopy := make(map[int]AttributeDomain)
		for k, v := range domains {
			domainsCopy[k] = AttributeDomain{Min: v.Min, Max: v.Max}
		}
		return domainsCopy
	}

	// Track constraints for each node
	domainConstraints := make([]map[int]AttributeDomain, len(nodes))
	for nodeIndex := range domainConstraints {
		domainConstraints[nodeIndex] = copyDomains(domains)
	}

	// Build tree from top down
	for nodeIndex := 0; nodeIndex < numSplits; nodeIndex++ {
		if len(domains) == 0 {
			// No more attributes to split on, implementation cannot handle this case
			log.Fatal("No more attributes to split on")
		}
		
		// Select an attribute
		attributeKey := int(randGen.Intn(len(domains), ))
		attrDomain := domainConstraints[nodeIndex][attributeKey]

		// Generate threshold within constrained domain
		if attrDomain.Max <= attrDomain.Min {
			// TODO: Adapt?
			log.Fatal("Invalid domain for attribute, cannot split further")
		}

		threshold := attrDomain.Min + randGen.Float64()*(attrDomain.Max-attrDomain.Min)
		nodes[nodeIndex] = Node{
			AttributeIndex: attributeKey,
			Threshold:      threshold,
			IsLeaf:         false,
		}

		// Propagate new constraints to children
		leftChildIndex, rightChildIndex := getChildIndices(nodeIndex)

		if leftChildIndex < len(nodes) {
			domainConstraints[leftChildIndex] = copyDomains(domainConstraints[nodeIndex])
			if attrDomain.Min == threshold {
				// remove attribute domain for left child
				delete(domainConstraints[leftChildIndex], attributeKey)
			} else {
				domainConstraints[leftChildIndex][attributeKey] = AttributeDomain{
					Min: attrDomain.Min,
					Max: threshold,
				}
			}
		}
		if rightChildIndex < len(nodes) {
			domainConstraints[rightChildIndex] = copyDomains(domainConstraints[nodeIndex])
			if attrDomain.Max == threshold {
				// remove attribute domain for right child
				delete(domainConstraints[rightChildIndex], attributeKey)
			} else {
				domainConstraints[rightChildIndex][attributeKey] = AttributeDomain{
					Min: threshold,
					Max: attrDomain.Max,
				}
			}
		}
	}



	// Add leaf nodes
	for i := numSplits; i < len(nodes); i++ {
		nodes[i] = Node{
			AttributeIndex: -1,
			Threshold:      -1,
			IsLeaf:         true,
		}
	}

	return PerfectBinaryTree{Nodes: nodes, Height: height}
}

func ReadAttributeDomains(filename string) AttributeDomains {

	data, err := os.ReadFile(filename)
	if err != nil {
		log.Fatalf("error reading file: %v", err)
	}

	attributeDomains := make(AttributeDomains)
	err = json.Unmarshal(data, &attributeDomains)
	if err != nil {
		log.Fatalf("error unmarshalling JSON: %v", err)
	}
	return attributeDomains
}

func (ad *AttributeDomains) UnmarshalJSON(data []byte) error {
	// Temp map with string keys and [2]float64 values
	var raw map[string][2]float64
	if err := json.Unmarshal(data, &raw); err != nil {
		return err
	}

	result := make(AttributeDomains)

	// Convert string keys to int, and array to struct
	for k, v := range raw {
		intKey, err := strconv.Atoi(k)
		if err != nil {
			return err
		}
		result[intKey] = AttributeDomain{
			Min: v[0],
			Max: v[1],
		}
	}

	*ad = result
	return nil
}

func CalculateLeafVector(trees []PerfectBinaryTree, records [][]float64, nonParticipatingProb float64) []uint64 {
	fmt.Println("Calculating leaf vector with trees", trees)
	numLeaves := 1 << trees[0].Height // We expect all trees to have the same height
	leafVector := make([]uint64, numLeaves * 2 * len(trees))

	for i, tree := range trees {

		for _, record := range records {
			// Simulate non-participation
			// if rand.Float64() < nonParticipatingProb {
			// 	fmt.Println("Node skipped record", i, "due to non-participation probability")
			// 	continue
			// }
			nodeIndex := 0
			for !tree.Nodes[nodeIndex].IsLeaf {
				node := tree.Nodes[nodeIndex]
				if record[node.AttributeIndex] <= node.Threshold {
					nodeIndex = 2*nodeIndex + 1 // go left
				} else {
					nodeIndex = 2*nodeIndex + 2 // go right
				}
			}

			leafStart := (1 << tree.Height) - 1
			leafOffset := nodeIndex - leafStart
			if leafOffset < 0 || leafOffset >= len(leafVector) {
				log.Fatalf("Leaf index out of bounds: %d", leafOffset)
			}

			// class is stored in last element of record
			recordClass := int(record[len(record)-1])
			vecIndex := leafOffset*2 + recordClass
			leafVector[i*numLeaves*2 + vecIndex]++
		}
	}

	// Randomly select nonParticipating Leaves based on the nonParticipationProb, create signal vector
	// TODO: Why is this not applied?
	numLeavesInTrees := len(trees) * numLeaves
	for i := 0; i < numLeavesInTrees; i++  {
		leafIndex := i * 2
		randFloat := rand.Float64()
		if randFloat < nonParticipatingProb {
			fmt.Println("random float", randFloat, "is less than nonParticipatingProb", nonParticipatingProb, "skipping leaf", i)
			leafVector[leafIndex] = 0 // Simulate non-participation by setting the count to 0
			leafVector[leafIndex+1] = 0 // Set both classes to 0
		}
	}
	return leafVector
}

func extractID(s string) (int, error) {
    re := regexp.MustCompile(`\d+`)
    match := re.FindString(s)
    return strconv.Atoi(match)
}

