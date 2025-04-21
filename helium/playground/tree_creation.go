package main

import (
	"encoding/json"
	"fmt"
	"math"
	"math/rand"
	"time"
)

type Node struct {
	AttributeIndex int
	Threshold      float64
	IsLeaf         bool
}

type PerfectBinaryTree struct {
	Nodes  []Node
	Height int
}

type AttributeDomain struct {
	Min float64
	Max float64
}

type AttributeDomains map[int]AttributeDomain

func CreateTreeStructure(domains AttributeDomains, height int) PerfectBinaryTree {
	numNodes := int(math.Pow(2, float64(height))) - 1
	nodes := make([]Node, numNodes)

	if numNodes == 0 {
		return PerfectBinaryTree{Nodes: nodes, Height: height}
	}

	for nodeIndex := 0; nodeIndex < numNodes; nodeIndex++ {
		attributeKey := int(rand.Intn(len(domains))) //Attribute keys are a continuous range of integers starting from 0
		attributeDomain := domains[attributeKey]
		threshold := attributeDomain.Min + rand.Float64()*(attributeDomain.Max-attributeDomain.Min)
		nodes[nodeIndex] = Node{
			AttributeIndex: attributeKey,
			Threshold:      threshold,
			IsLeaf:         false,//TODO: Implement leaf nodes
		}
	}

	return PerfectBinaryTree{Nodes: nodes, Height: height}
}

func main() {
	rand.New(rand.NewSource(time.Now().UnixNano()))

	attributeDomains := AttributeDomains{
		0: {Min: 0.0, Max: 1.0},
		1: {Min: 5.0, Max: 10.0},
		2: {Min: 0.0, Max: 2.0},
	}

	height := 3
	tree := CreateTreeStructure(attributeDomains, height)

	fmt.Printf("Created a perfect binary tree (iterative) with %d nodes and height %d\n", len(tree.Nodes), tree.Height)
	for i, node := range tree.Nodes {
		fmt.Printf("Node %d: AttributeIndex=%d, Threshold=%.2f, IsLeaf=%t\n", i, node.AttributeIndex, node.Threshold, node.IsLeaf)
	}

	height = 4
	tree = CreateTreeStructure(attributeDomains, height)

	fmt.Printf("\nCreated a perfect binary tree (iterative) with %d nodes and height %d\n", len(tree.Nodes), tree.Height)
	for i, node := range tree.Nodes {
		fmt.Printf("Node %d: AttributeIndex=%d, Threshold=%.2f, IsLeaf=%t\n", i, node.AttributeIndex, node.Threshold, node.IsLeaf)
	}

    jsonStr, _ := json.Marshal(tree) // handle errors properly in real code
    var decoded PerfectBinaryTree
	json.Unmarshal([]byte(jsonStr), &decoded)
	fmt.Printf("\nDecoded tree: %v\n", decoded)
	for i, node := range decoded.Nodes {
		fmt.Printf("Node %d: AttributeIndex=%d, Threshold=%.2f, IsLeaf=%t\n", i, node.AttributeIndex, node.Threshold, node.IsLeaf)
	}

}