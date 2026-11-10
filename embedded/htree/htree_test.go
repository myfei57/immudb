/*
Copyright 2026 Codenotary Inc. All rights reserved.

SPDX-License-Identifier: BUSL-1.1
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://mariadb.com/bsl11/

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package htree

import (
	"crypto/sha256"
	"encoding/binary"
	"math"
	"math/bits"
	"math/rand"
	"sort"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestHTree(t *testing.T) {
	const maxWidth = 1000

	tree, err := New(0)
	require.NoError(t, err)

	err = tree.BuildWith([][sha256.Size]byte{sha256.Sum256(nil)})
	require.ErrorIs(t, err, ErrMaxWidthExceeded)

	err = tree.BuildWith(nil)
	require.NoError(t, err)
	require.Equal(t, sha256.Sum256(nil), tree.Root())

	tree, err = New(maxWidth)
	require.NoError(t, err)

	digests := make([][sha256.Size]byte, maxWidth)

	for i := 0; i < len(digests); i++ {
		var b [8]byte
		binary.BigEndian.PutUint64(b[:], uint64(i))
		digests[i] = sha256.Sum256(b[:])
	}

	err = tree.BuildWith(digests)
	require.NoError(t, err)

	root := tree.Root()

	for i := 0; i < len(digests); i++ {
		proof, err := tree.InclusionProof(i)
		require.NoError(t, err)
		require.NotNil(t, proof)

		verifies := VerifyInclusion(proof, digests[i], root)
		require.True(t, verifies)

		verifies = VerifyInclusion(proof, sha256.Sum256(digests[i][:]), root)
		require.False(t, verifies)

		verifies = VerifyInclusion(proof, digests[i], sha256.Sum256(root[:]))
		require.False(t, verifies)

		proof.Terms = nil
		verifies = VerifyInclusion(proof, digests[i], root)
		require.False(t, verifies)

		verifies = VerifyInclusion(nil, digests[i], root)
		require.False(t, verifies)
	}

	err = tree.BuildWith(nil)
	require.NoError(t, err)
	require.Equal(t, sha256.Sum256(nil), tree.Root())

	err = tree.BuildWith(make([][sha256.Size]byte, maxWidth+1))
	require.ErrorIs(t, err, ErrMaxWidthExceeded)

	_, err = tree.InclusionProof(maxWidth)
	require.ErrorIs(t, err, ErrIllegalArguments)
}

func testDigests(n int) [][sha256.Size]byte {
	digests := make([][sha256.Size]byte, n)

	for i := 0; i < n; i++ {
		var b [8]byte
		binary.BigEndian.PutUint64(b[:], uint64(i))
		digests[i] = sha256.Sum256(b[:])
	}

	return digests
}

func pickDigests(leaves []int, digests [][sha256.Size]byte) [][sha256.Size]byte {
	picked := make([][sha256.Size]byte, len(leaves))

	for i, leaf := range leaves {
		picked[i] = digests[leaf]
	}

	return picked
}

func subsetFromMask(width, mask int) []int {
	leaves := make([]int, 0, width)

	for i := 0; i < width; i++ {
		if mask&(1<<i) != 0 {
			leaves = append(leaves, i)
		}
	}

	return leaves
}

// requireTermsAreRealNodes checks that every term addresses an actual node of
// the tree, terms are canonically ordered and their coverage is disjoint and
// never overlaps a requested leaf.
func requireTermsAreRealNodes(
	t *testing.T,
	tree *HTree,
	proof *MultiInclusionProof,
	leaves []int,
) {
	t.Helper()

	targets := make(map[int]bool, len(leaves))
	for _, leaf := range leaves {
		targets[leaf] = true
	}

	covered := make(map[int]bool)
	prevStart := -1

	for _, term := range proof.Terms {
		require.Equal(t, tree.levels[term.Layer][term.Index], term.Hash)

		start, end := termCoverage(proof.Width, term.Layer, term.Index)
		require.Greater(t, start, prevStart, "terms must be canonically ordered")

		for idx := start; idx < end; idx++ {
			require.False(t, covered[idx], "term coverage must be disjoint")
			require.False(t, targets[idx], "term must never cover a requested leaf")
			covered[idx] = true
		}

		prevStart = start
	}
}

func singleProofTermHashes(t *testing.T, tree *HTree, leaf int) [][sha256.Size]byte {
	t.Helper()

	proof, err := tree.InclusionProof(leaf)
	require.NoError(t, err)

	return proof.Terms
}

// TestMultiInclusionProofExhaustive verifies every non-empty leaf subset of
// every small width, including non-power-of-two widths, against both the
// multi verifier and the existing per-leaf InclusionProof.
func TestMultiInclusionProofExhaustive(t *testing.T) {
	const maxWidth = 8

	for width := 1; width <= maxWidth; width++ {
		tree, err := New(width)
		require.NoError(t, err)

		digests := testDigests(width)
		require.NoError(t, tree.BuildWith(digests))

		root := tree.Root()

		for mask := 1; mask < (1 << width); mask++ {
			leaves := subsetFromMask(width, mask)
			picked := pickDigests(leaves, digests)

			proof, err := tree.MultiInclusionProof(leaves)
			require.NoError(t, err)
			require.NotNil(t, proof)
			require.Equal(t, width, proof.Width)
			require.Equal(t, leaves, proof.Leaves)
			require.Less(t, len(proof.Terms), width)

			require.True(t, VerifyMultiInclusion(proof, width, leaves, picked, root))

			// Cross-check with the existing single-leaf verification.
			for _, leaf := range leaves {
				single, err := tree.InclusionProof(leaf)
				require.NoError(t, err)
				require.True(t, VerifyInclusion(single, digests[leaf], root))
			}

			requireTermsAreRealNodes(t, tree, proof, leaves)

			// The shared proof never needs more auxiliary hashes than the
			// concatenation of the individual per-leaf proofs.
			concatTerms := 0
			for _, leaf := range leaves {
				concatTerms += len(singleProofTermHashes(t, tree, leaf))
			}
			require.LessOrEqual(t, len(proof.Terms), concatTerms)

			// Any digest or root mismatch must invalidate the proof.
			tampered := make([][sha256.Size]byte, len(picked))
			copy(tampered, picked)
			tampered[0][0] ^= 0xFF
			require.False(t, VerifyMultiInclusion(proof, width, leaves, tampered, root))

			wrongRoot := root
			wrongRoot[0] ^= 0xFF
			require.False(t, VerifyMultiInclusion(proof, width, leaves, picked, wrongRoot))

			// The verifier binds the expected width, not only proof.Width.
			require.False(t, VerifyMultiInclusion(proof, width+1, leaves, picked, root))
		}
	}
}

// TestMultiInclusionProofSharing checks that shared paths are actually merged
// instead of concatenating per-leaf InclusionProof terms.
func TestMultiInclusionProofSharing(t *testing.T) {
	// In a width-4 tree, adjacent leaves 0 and 1 only need the sibling subtree
	// rooted at (layer 1, index 1), i.e. a single auxiliary hash, while the two
	// single proofs carry four terms in total.
	tree4, err := New(4)
	require.NoError(t, err)

	digests4 := testDigests(4)
	require.NoError(t, tree4.BuildWith(digests4))

	proof, err := tree4.MultiInclusionProof([]int{0, 1})
	require.NoError(t, err)
	require.Len(t, proof.Terms, 1)
	require.Equal(t, MultiProofTerm{
		Layer: 1,
		Index: 1,
		Hash:  tree4.levels[1][1],
	}, proof.Terms[0])
	require.True(t, VerifyMultiInclusion(proof, 4, []int{0, 1}, pickDigests([]int{0, 1}, digests4), tree4.Root()))

	concatTerms := len(singleProofTermHashes(t, tree4, 0)) + len(singleProofTermHashes(t, tree4, 1))
	require.Equal(t, 4, concatTerms)

	tree, err := New(7)
	require.NoError(t, err)

	digests := testDigests(7)
	require.NoError(t, tree.BuildWith(digests))
	root := tree.Root()

	// Requesting all leaves requires no auxiliary hash at all.
	for width := 1; width <= 7; width++ {
		sub := tree
		allLeaves := make([]int, width)
		for i := range allLeaves {
			allLeaves[i] = i
		}

		if width != tree.width {
			sub, err = New(width)
			require.NoError(t, err)
			require.NoError(t, sub.BuildWith(testDigests(width)))
		}

		allProof, err := sub.MultiInclusionProof(allLeaves)
		require.NoError(t, err)
		require.Empty(t, allProof.Terms)
		require.True(t, VerifyMultiInclusion(allProof, width, allLeaves, testDigests(width), sub.Root()))
	}

	// A single-leaf multi proof carries exactly the single proof hashes.
	for leaf := 0; leaf < 7; leaf++ {
		multi, err := tree.MultiInclusionProof([]int{leaf})
		require.NoError(t, err)

		expected := singleProofTermHashes(t, tree, leaf)
		require.Len(t, multi.Terms, len(expected))

		gotHashes := make(map[[sha256.Size]byte]bool, len(multi.Terms))
		for _, term := range multi.Terms {
			gotHashes[term.Hash] = true
		}

		for _, hash := range expected {
			require.True(t, gotHashes[hash])
		}
	}

	// A wider shared case must strictly reduce the auxiliary hash count.
	shared, err := tree.MultiInclusionProof([]int{0, 1, 2})
	require.NoError(t, err)
	require.True(t, VerifyMultiInclusion(shared, 7, []int{0, 1, 2}, pickDigests([]int{0, 1, 2}, digests), root))

	concat := 0
	for _, leaf := range []int{0, 1, 2} {
		concat += len(singleProofTermHashes(t, tree, leaf))
	}
	require.Less(t, len(shared.Terms), concat)
}

// TestMultiInclusionProofLeafPrefix ensures digests are still hashed exactly
// once with the leaf prefix, matching BuildWith semantics.
func TestMultiInclusionProofLeafPrefix(t *testing.T) {
	digest := sha256.Sum256([]byte("single-digest"))

	tree, err := New(1)
	require.NoError(t, err)
	require.NoError(t, tree.BuildWith([][sha256.Size]byte{digest}))

	leaf := [1 + sha256.Size]byte{LeafPrefix}
	copy(leaf[1:], digest[:])
	expectedRoot := sha256.Sum256(leaf[:])
	require.Equal(t, expectedRoot, tree.Root())

	proof, err := tree.MultiInclusionProof([]int{0})
	require.NoError(t, err)
	require.Empty(t, proof.Terms)
	require.True(t, VerifyMultiInclusion(proof, 1, []int{0}, [][sha256.Size]byte{digest}, expectedRoot))
}

func TestMultiInclusionProofInputValidation(t *testing.T) {
	tree, err := New(5)
	require.NoError(t, err)
	require.NoError(t, tree.BuildWith(testDigests(5)))

	for _, leaves := range [][]int{
		nil,
		{},
		{5},
		{-1},
		{0, 5},
		{2, 1},
		{1, 1},
		{1, 3, 3},
	} {
		_, err := tree.MultiInclusionProof(leaves)
		require.ErrorIs(t, err, ErrIllegalArguments)
	}

	// A tree with no leaves rejects every request.
	emptyTree, err := New(5)
	require.NoError(t, err)
	require.NoError(t, emptyTree.BuildWith(nil))

	_, err = emptyTree.MultiInclusionProof([]int{0})
	require.ErrorIs(t, err, ErrIllegalArguments)

	root := tree.Root()
	digests := testDigests(5)
	valid, err := tree.MultiInclusionProof([]int{1, 3})
	require.NoError(t, err)

	require.False(t, VerifyMultiInclusion(nil, 5, []int{1, 3}, pickDigests([]int{1, 3}, digests), root))
	require.False(t, VerifyMultiInclusion(valid, 5, nil, pickDigests([]int{1, 3}, digests), root))
	require.False(t, VerifyMultiInclusion(valid, 5, []int{1}, pickDigests([]int{1, 3}, digests), root))
	require.False(t, VerifyMultiInclusion(valid, 5, []int{1, 3}, [][sha256.Size]byte{digests[1]}, root))
	require.False(t, VerifyMultiInclusion(valid, 5, []int{1, 2}, pickDigests([]int{1, 3}, digests), root))
	require.False(t, VerifyMultiInclusion(valid, 5, []int{3, 1}, pickDigests([]int{1, 3}, digests), root))

	// The expected width is bound independently of the proof content.
	require.False(t, VerifyMultiInclusion(valid, 4, []int{1, 3}, pickDigests([]int{1, 3}, digests), root))
	require.False(t, VerifyMultiInclusion(valid, 6, []int{1, 3}, pickDigests([]int{1, 3}, digests), root))

	// Malicious or overflowing widths are rejected before any allocation.
	for _, width := range []int{0, -1, math.MaxInt, math.MaxInt32, 1 << 40} {
		malicious := *valid
		malicious.Width = width
		require.False(t, VerifyMultiInclusion(&malicious, width, []int{1, 3}, pickDigests([]int{1, 3}, digests), root))
		require.False(t, VerifyMultiInclusion(&malicious, 5, []int{1, 3}, pickDigests([]int{1, 3}, digests), root))
	}
}

func cloneMultiProof(proof *MultiInclusionProof) *MultiInclusionProof {
	clone := &MultiInclusionProof{
		Width:  proof.Width,
		Leaves: append([]int(nil), proof.Leaves...),
		Terms:  append([]MultiProofTerm(nil), proof.Terms...),
	}
	return clone
}

// TestVerifyMultiInclusionTamper randomly mutates every field of otherwise
// valid proofs and requires the verifier to reject all of them. Only
// applicable mutations are selected, so every iteration changes the proof.
func TestVerifyMultiInclusionTamper(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	digests := testDigests(12)

	const iterations = 2000

	for i := 0; i < iterations; i++ {
		width := 1 + rng.Intn(12)
		mask := 1 + rng.Intn((1<<width)-1)
		leaves := subsetFromMask(width, mask)
		picked := pickDigests(leaves, digests[:width])

		tree, err := New(width)
		require.NoError(t, err)
		require.NoError(t, tree.BuildWith(digests[:width]))
		root := tree.Root()

		valid, err := tree.MultiInclusionProof(leaves)
		require.NoError(t, err)
		require.True(t, VerifyMultiInclusion(valid, width, leaves, picked, root))

		mutated := cloneMultiProof(valid)
		mutatedDigests := append([][sha256.Size]byte(nil), picked...)
		mutatedLeaves := append([]int(nil), leaves...)
		mutatedRoot := root

		type mutation struct {
			apply func()
		}

		mutations := []mutation{
			{func() {
				mutatedDigests[rng.Intn(len(mutatedDigests))][rng.Intn(sha256.Size)] ^= 1 << rng.Intn(8)
			}},
			{func() {
				mutatedRoot[rng.Intn(sha256.Size)] ^= 1 << rng.Intn(8)
			}},
			{func() {
				mutated.Width += 1 + rng.Intn(3)
			}},
			{func() {
				mutated.Leaves[rng.Intn(len(mutated.Leaves))] = width
			}},
			{func() {
				junk := MultiProofTerm{Layer: 0, Index: leaves[0]}
				junk.Hash[rng.Intn(sha256.Size)] = 0xFF
				mutated.Terms = append(mutated.Terms, junk)
			}},
		}

		if width > 1 {
			mutations = append(mutations, mutation{func() {
				pos := rng.Intn(len(mutatedLeaves))
				replacement := mutatedLeaves[pos]
				for replacement == mutatedLeaves[pos] {
					replacement = rng.Intn(width)
				}
				mutatedLeaves[pos] = replacement
			}})
		}

		if len(mutated.Terms) > 0 {
			mutations = append(mutations, mutation{func() {
				pos := rng.Intn(len(mutated.Terms))
				mutated.Terms[pos].Hash[rng.Intn(sha256.Size)] ^= 1 << rng.Intn(8)
			}})
			mutations = append(mutations, mutation{func() {
				pos := rng.Intn(len(mutated.Terms))
				original := mutated.Terms[pos]

				// Promoted odd-tail nodes share their digest with lower
				// layers, so keep mutating until the new coordinate is out of
				// bounds or addresses a node with a different digest.
				topLayer := bits.Len(uint(width - 1))

				for {
					newLayer := rng.Intn(bits.Len(uint(width)) + 1)
					newIndex := rng.Intn(width)

					if newLayer == original.Layer && newIndex == original.Index {
						continue
					}

					if newLayer > topLayer || newIndex >= layerNodeCount(width, newLayer) {
						mutated.Terms[pos].Layer = newLayer
						mutated.Terms[pos].Index = newIndex
						return
					}

					if tree.levels[newLayer][newIndex] != original.Hash {
						mutated.Terms[pos].Layer = newLayer
						mutated.Terms[pos].Index = newIndex
						return
					}
				}
			}})
			mutations = append(mutations, mutation{func() {
				pos := rng.Intn(len(mutated.Terms))
				mutated.Terms = append(mutated.Terms, mutated.Terms[pos])
			}})
			mutations = append(mutations, mutation{func() {
				pos := rng.Intn(len(mutated.Terms))
				mutated.Terms = append(mutated.Terms[:pos], mutated.Terms[pos+1:]...)
			}})
		}

		if len(mutated.Terms) > 1 {
			mutations = append(mutations, mutation{func() {
				a, b := rng.Intn(len(mutated.Terms)), rng.Intn(len(mutated.Terms))
				for a == b {
					b = rng.Intn(len(mutated.Terms))
				}
				mutated.Terms[a], mutated.Terms[b] = mutated.Terms[b], mutated.Terms[a]
			}})
		}

		mutations[rng.Intn(len(mutations))].apply()

		require.False(t,
			VerifyMultiInclusion(mutated, width, mutatedLeaves, mutatedDigests, mutatedRoot),
			"iteration %d accepted tampered proof", i)
	}
}

// TestMultiInclusionProofIndependentMemory ensures issued proofs own their
// memory and survive later BuildWith calls, and that proof generation never
// changes the tree root.
func TestMultiInclusionProofIndependentMemory(t *testing.T) {
	tree, err := New(6)
	require.NoError(t, err)

	digests := testDigests(6)
	require.NoError(t, tree.BuildWith(digests))
	rootA := tree.Root()

	leaves := []int{0, 2, 5}
	proof, err := tree.MultiInclusionProof(leaves)
	require.NoError(t, err)

	// Issuing a proof does not change the root.
	require.Equal(t, rootA, tree.Root())

	// Mutating the caller's slices does not affect the proof.
	leaves[0] = 4
	require.Equal(t, []int{0, 2, 5}, proof.Leaves)

	// Rebuild the same tree with different digests.
	otherDigests := make([][sha256.Size]byte, 6)
	for i := range otherDigests {
		otherDigests[i] = sha256.Sum256([]byte{byte(i), byte(i + 1)})
	}
	require.NoError(t, tree.BuildWith(otherDigests))
	rootB := tree.Root()
	require.NotEqual(t, rootA, rootB)

	// The old proof remains valid against the old root and fails against the
	// new one, proving it holds independent memory.
	require.True(t, VerifyMultiInclusion(proof, 6, []int{0, 2, 5}, pickDigests([]int{0, 2, 5}, digests), rootA))
	require.False(t, VerifyMultiInclusion(proof, 6, []int{0, 2, 5}, pickDigests([]int{0, 2, 5}, digests), rootB))
}

// TestMultiInclusionProofRejectsInconsistentStructure exercises structural
// rejection paths that random tampering may not cover.
func TestMultiInclusionProofRejectsInconsistentStructure(t *testing.T) {
	tree, err := New(5)
	require.NoError(t, err)

	digests := testDigests(5)
	require.NoError(t, tree.BuildWith(digests))
	root := tree.Root()

	valid, err := tree.MultiInclusionProof([]int{0, 4})
	require.NoError(t, err)
	require.True(t, VerifyMultiInclusion(valid, 5, []int{0, 4}, pickDigests([]int{0, 4}, digests), root))

	// Out-of-bounds term coordinate.
	outOfBounds := cloneMultiProof(valid)
	if len(outOfBounds.Terms) > 0 {
		outOfBounds.Terms[0].Index = layerNodeCount(outOfBounds.Width, outOfBounds.Terms[0].Layer)
		require.False(t, VerifyMultiInclusion(outOfBounds, 5, []int{0, 4}, pickDigests([]int{0, 4}, digests), root))
	}

	// Term covering a requested leaf is rejected.
	overlapping := cloneMultiProof(valid)
	overlapping.Terms = append([]MultiProofTerm{{Layer: 0, Index: 0, Hash: tree.levels[0][0]}}, overlapping.Terms...)
	require.False(t, VerifyMultiInclusion(overlapping, 5, []int{0, 4}, pickDigests([]int{0, 4}, digests), root))

	// Duplicated term is rejected.
	if len(valid.Terms) > 0 {
		duplicated := cloneMultiProof(valid)
		duplicated.Terms = append(duplicated.Terms, duplicated.Terms[0])
		require.False(t, VerifyMultiInclusion(duplicated, 5, []int{0, 4}, pickDigests([]int{0, 4}, digests), root))
	}

	// A digest aligned to the wrong leaf is rejected.
	swapped := pickDigests([]int{0, 4}, digests)
	swapped[0], swapped[1] = swapped[1], swapped[0]
	require.False(t, VerifyMultiInclusion(valid, 5, []int{0, 4}, swapped, root))
}

// TestMultiInclusionProofLargerWidths exercises power-of-two and odd widths
// beyond the exhaustive range, including the width used by TestHTree.
func TestMultiInclusionProofLargerWidths(t *testing.T) {
	rng := rand.New(rand.NewSource(7))

	for _, width := range []int{15, 16, 17, 31, 32, 33, 100, 255, 1000} {
		tree, err := New(width)
		require.NoError(t, err)

		digests := testDigests(width)
		require.NoError(t, tree.BuildWith(digests))
		root := tree.Root()

		for iteration := 0; iteration < 40; iteration++ {
			subsetSize := 1 + rng.Intn(width)
			leaves := rng.Perm(width)[:subsetSize]
			sort.Ints(leaves)
			picked := pickDigests(leaves, digests)

			proof, err := tree.MultiInclusionProof(leaves)
			require.NoError(t, err)
			require.Less(t, len(proof.Terms), width)
			require.True(t, VerifyMultiInclusion(proof, width, leaves, picked, root))
			requireTermsAreRealNodes(t, tree, proof, leaves)

			concatTerms := 0
			for _, leaf := range leaves {
				concatTerms += len(singleProofTermHashes(t, tree, leaf))
			}
			require.LessOrEqual(t, len(proof.Terms), concatTerms)
		}

		// The full leaf set always verifies with no auxiliary term.
		allLeaves := make([]int, width)
		for i := range allLeaves {
			allLeaves[i] = i
		}

		allProof, err := tree.MultiInclusionProof(allLeaves)
		require.NoError(t, err)
		require.Empty(t, allProof.Terms)
		require.True(t, VerifyMultiInclusion(allProof, width, allLeaves, digests, root))
	}
}

// TestMultiInclusionProofPromotedCoordinates documents that coordinates
// linked by the odd-tail promotion rule carry the same digest, so a term may
// be expressed at any of them while proving exactly the same leaf range.
func TestMultiInclusionProofPromotedCoordinates(t *testing.T) {
	tree, err := New(5)
	require.NoError(t, err)

	digests := testDigests(5)
	require.NoError(t, tree.BuildWith(digests))
	root := tree.Root()

	// Leaf 4 is promoted unchanged through layers 0, 1 and 2.
	require.Equal(t, tree.levels[0][4], tree.levels[1][2])
	require.Equal(t, tree.levels[1][2], tree.levels[2][1])

	proof, err := tree.MultiInclusionProof([]int{0})
	require.NoError(t, err)

	// The proof addresses leaf 4 directly at layer 0.
	var termPos int
	var found bool

	for pos, term := range proof.Terms {
		if term.Layer == 0 && term.Index == 4 {
			termPos = pos
			found = true
		}
	}
	require.True(t, found)
	require.True(t, VerifyMultiInclusion(proof, 5, []int{0}, [][sha256.Size]byte{digests[0]}, root))

	// Expressing the same auxiliary digest at the promoted coordinate covers
	// the same leaf range and must still verify.
	promoted := cloneMultiProof(proof)
	promoted.Terms[termPos] = MultiProofTerm{Layer: 1, Index: 2, Hash: tree.levels[1][2]}
	require.True(t, VerifyMultiInclusion(promoted, 5, []int{0}, [][sha256.Size]byte{digests[0]}, root))

	// A different digest at that coordinate is rejected.
	forged := cloneMultiProof(promoted)
	forged.Terms[termPos].Hash[0] ^= 0xFF
	require.False(t, VerifyMultiInclusion(forged, 5, []int{0}, [][sha256.Size]byte{digests[0]}, root))
}
