package hnsw

import (
	"cmp"
	"math/rand"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"
)

func Test_maxLevel(t *testing.T) {
	var m int

	m = maxLevel(0.5, 10)
	require.Equal(t, 4, m)

	m = maxLevel(0.5, 1000)
	require.Equal(t, 11, m)
}

func Test_layerNode_search(t *testing.T) {
	entry := &layerNode[int]{
		Node: Node[int]{
			Value: Vector{0},
			Key:   0,
		},
		neighbors: map[int]*layerNode[int]{
			1: {
				Node: Node[int]{
					Value: Vector{1},
					Key:   1,
				},
			},
			2: {
				Node: Node[int]{
					Value: Vector{2},
					Key:   2,
				},
			},
			3: {
				Node: Node[int]{
					Value: Vector{3},
					Key:   3,
				},
				neighbors: map[int]*layerNode[int]{
					4: {
						Node: Node[int]{
							Value: Vector{4},
							Key:   5,
						},
					},
					5: {
						Node: Node[int]{
							Value: Vector{5},
							Key:   5,
						},
					},
				},
			},
		},
	}

	best := entry.search(2, 4, []float32{4}, EuclideanDistance)

	require.Equal(t, 5, best[0].node.Key)
	require.Equal(t, 3, best[1].node.Key)
	require.Len(t, best, 2)
}

func newTestGraph[K cmp.Ordered]() *Graph[K] {
	return &Graph[K]{
		M:        6,
		Distance: EuclideanDistance,
		Ml:       0.5,
		EfSearch: 20,
		Rng:      rand.New(rand.NewSource(0)),
	}
}

func TestGraph_AddSearch(t *testing.T) {
	t.Parallel()

	g := newTestGraph[int]()

	for i := 0; i < 128; i++ {
		g.Add(
			Node[int]{
				Key:   i,
				Value: Vector{float32(i)},
			},
		)
	}

	al := Analyzer[int]{Graph: g}

	// Layers should be approximately log2(128) = 7
	// Look for an approximate doubling of the number of nodes in each layer.
	require.Equal(t, []int{
		128,
		67,
		28,
		12,
		6,
		2,
		1,
		1,
	}, al.Topography())

	nearest := g.Search(
		[]float32{64.5},
		4,
	)

	require.Len(t, nearest, 4)
	require.EqualValues(
		t,
		[]Node[int]{
			{64, Vector{64}},
			{65, Vector{65}},
			{62, Vector{62}},
			{63, Vector{63}},
		},
		nearest,
	)
}

func TestGraph_AddDelete(t *testing.T) {
	t.Parallel()

	g := newTestGraph[int]()
	for i := 0; i < 128; i++ {
		g.Add(Node[int]{
			Key:   i,
			Value: Vector{float32(i)},
		})
	}

	require.Equal(t, 128, g.Len())
	an := Analyzer[int]{Graph: g}

	preDeleteConnectivity := an.Connectivity()

	// Delete every even node.
	for i := 0; i < 128; i += 2 {
		ok := g.Delete(i)
		require.True(t, ok)
	}

	require.Equal(t, 64, g.Len())

	postDeleteConnectivity := an.Connectivity()

	// Connectivity may decrease after deletion since all references
	// to deleted nodes are properly cleaned up (including unidirectional
	// edges). It should remain well above half the original connectivity.
	require.Greater(
		t, postDeleteConnectivity[0],
		preDeleteConnectivity[0]*0.5,
	)

	t.Run("DeleteNotFound", func(t *testing.T) {
		ok := g.Delete(-1)
		require.False(t, ok)
	})
}

func Benchmark_HSNW(b *testing.B) {
	b.ReportAllocs()

	sizes := []int{100, 1000, 10000}

	// Use this to ensure that complexity is O(log n) where n = h.Len().
	for _, size := range sizes {
		b.Run(strconv.Itoa(size), func(b *testing.B) {
			g := Graph[int]{}
			g.Ml = 0.5
			g.Distance = EuclideanDistance
			for i := 0; i < size; i++ {
				g.Add(Node[int]{
					Key:   i,
					Value: Vector{float32(i)},
				})
			}
			b.ResetTimer()

			b.Run("Search", func(b *testing.B) {
				for i := 0; i < b.N; i++ {
					g.Search(
						[]float32{float32(i % size)},
						4,
					)
				}
			})
		})
	}
}

func randFloats(n int) []float32 {
	x := make([]float32, n)
	for i := range x {
		x[i] = rand.Float32()
	}
	return x
}

func Benchmark_HNSW_1536(b *testing.B) {
	b.ReportAllocs()

	g := newTestGraph[int]()
	const size = 1000
	points := make([]Node[int], size)
	for i := 0; i < size; i++ {
		points[i] = Node[int]{
			Key:   i,
			Value: Vector(randFloats(1536)),
		}
		g.Add(points[i])
	}
	b.ResetTimer()

	b.Run("Search", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			g.Search(
				points[i%size].Value,
				4,
			)
		}
	})
}

func TestGraph_DefaultCosine(t *testing.T) {
	g := NewGraph[int]()
	g.Add(
		Node[int]{Key: 1, Value: Vector{1, 1}},
		Node[int]{Key: 2, Value: Vector{0, 1}},
		Node[int]{Key: 3, Value: Vector{1, -1}},
	)

	neighbors := g.Search(
		[]float32{0.5, 0.5},
		1,
	)

	require.Equal(
		t,
		[]Node[int]{
			{1, Vector{1, 1}},
		},
		neighbors,
	)
}

// TestGraph_DeleteThenAdd_NoPanic reproduces a bug where Delete leaves stale
// neighbor pointers in the graph, causing a nil pointer panic on subsequent Add.
// See: https://github.com/heyajulia/aria/issues/31
func TestGraph_DeleteThenAdd_NoPanic(t *testing.T) {
	t.Parallel()

	const dim = 16

	for seed := int64(0); seed < 100; seed++ {
		rng := rand.New(rand.NewSource(seed))
		g := &Graph[int]{
			M:        6,
			Ml:       0.25,
			Distance: CosineDistance,
			EfSearch: 20,
			Rng:      rand.New(rand.NewSource(seed)),
		}

		// Add initial nodes.
		for i := 0; i < 100; i++ {
			vec := make(Vector, dim)
			for j := range vec {
				vec[j] = rng.Float32()*2 - 1
			}
			g.Add(MakeNode(i, vec))
		}

		// Delete a batch of nodes — this can corrupt the graph.
		for i := 0; i < 20; i++ {
			g.Delete(i * 5)
		}

		// Adding new nodes must not panic due to stale neighbor pointers.
		for i := 100; i < 200; i++ {
			vec := make(Vector, dim)
			for j := range vec {
				vec[j] = rng.Float32()*2 - 1
			}
			g.Add(MakeNode(i, vec))
		}
	}
}

// TestGraph_DeleteIntegrity validates that after deletion, no node in the graph
// has a neighbor pointer to a node that is not in the layer's node map.
func TestGraph_DeleteIntegrity(t *testing.T) {
	t.Parallel()

	const dim = 16

	for seed := int64(0); seed < 50; seed++ {
		rng := rand.New(rand.NewSource(seed))
		g := &Graph[int]{
			M:        6,
			Ml:       0.25,
			Distance: CosineDistance,
			EfSearch: 20,
			Rng:      rand.New(rand.NewSource(seed)),
		}

		for i := 0; i < 100; i++ {
			vec := make(Vector, dim)
			for j := range vec {
				vec[j] = rng.Float32()*2 - 1
			}
			g.Add(MakeNode(i, vec))
		}

		for i := 0; i < 20; i++ {
			g.Delete(i * 5)
		}

		// Validate that no node references a deleted node.
		for li, layer := range g.layers {
			for key, node := range layer.nodes {
				for nKey := range node.neighbors {
					if _, ok := layer.nodes[nKey]; !ok {
						t.Fatalf(
							"seed %d: layer %d: node %v has neighbor %v which is not in the layer's node map (dangling pointer)",
							seed, li, key, nKey,
						)
					}
				}
			}
		}
	}
}

// TestGraph_DeleteAddSearchCycle is a regression test that exercises
// add-delete-add-search cycles. The original bug caused a nil pointer
// panic when a deleted node's key was used as a search entry point
// (elevator) because stale neighbor pointers survived deletion.
func TestGraph_DeleteAddSearchCycle(t *testing.T) {
	t.Parallel()

	const dim = 8

	for seed := int64(0); seed < 50; seed++ {
		rng := rand.New(rand.NewSource(seed))
		g := &Graph[int]{
			M:        4,
			Ml:       0.25,
			Distance: EuclideanDistance,
			EfSearch: 10,
			Rng:      rand.New(rand.NewSource(seed)),
		}

		randVec := func() Vector {
			v := make(Vector, dim)
			for j := range v {
				v[j] = rng.Float32()*2 - 1
			}
			return v
		}

		// Phase 1: populate
		for i := 0; i < 50; i++ {
			g.Add(MakeNode(i, randVec()))
		}

		// Phase 2: delete a batch
		for i := 0; i < 50; i += 3 {
			g.Delete(i)
		}

		// Phase 3: add more nodes
		for i := 50; i < 100; i++ {
			g.Add(MakeNode(i, randVec()))
		}

		// Phase 4: search must not panic
		for i := 0; i < 10; i++ {
			results := g.Search(randVec(), 5)
			require.NotEmpty(t, results, "seed %d: search returned no results", seed)
		}

		// Phase 5: delete another batch
		for i := 50; i < 100; i += 2 {
			g.Delete(i)
		}

		// Phase 6: add and search again
		for i := 100; i < 150; i++ {
			g.Add(MakeNode(i, randVec()))
		}
		results := g.Search(randVec(), 3)
		require.NotEmpty(t, results, "seed %d: final search returned no results", seed)

		// Validate graph integrity after all operations.
		for li, layer := range g.layers {
			for key, node := range layer.nodes {
				for nKey := range node.neighbors {
					if _, ok := layer.nodes[nKey]; !ok {
						t.Fatalf(
							"seed %d: layer %d: node %v has dangling neighbor %v",
							seed, li, key, nKey,
						)
					}
				}
			}
		}
	}
}

func TestGraph_RemoveAllNodes(t *testing.T) {
	var vec = []float32{1}

	for i := 0; i < 10; i++ {
		g := NewGraph[int]()
		g.Add(MakeNode(1, vec))
		g.Delete(1)
		g.Add(MakeNode(1, vec))
	}
}
