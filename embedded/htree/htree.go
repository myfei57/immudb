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
	"errors"
	"math/bits"
	"sort"
)

var ErrMaxWidthExceeded = errors.New("htree: max width exceeded")
var ErrIllegalArguments = errors.New("htree: illegal arguments")
var ErrIllegalState = errors.New("htree: illegal state")

const LeafPrefix = byte(0)
const NodePrefix = byte(1)

type HTree struct {
	levels   [][][sha256.Size]byte
	maxWidth int
	width    int
	root     [sha256.Size]byte
}

type InclusionProof struct {
	Leaf  int
	Width int
	Terms [][sha256.Size]byte
}

func New(maxWidth int) (*HTree, error) {
	var levels [][][sha256.Size]byte

	if maxWidth > 0 {
		lw := 1
		for lw < maxWidth {
			lw = lw << 1
		}

		height := bits.Len64(uint64(maxWidth-1)) + 1

		levels = make([][][sha256.Size]byte, height)
		for l := 0; l < height; l++ {
			levels[l] = make([][sha256.Size]byte, lw>>l)
		}
	}

	return &HTree{
		levels:   levels,
		maxWidth: maxWidth,
	}, nil
}

func (t *HTree) BuildWith(digests [][sha256.Size]byte) error {
	if len(digests) > t.maxWidth {
		return ErrMaxWidthExceeded
	}

	if len(digests) == 0 {
		t.width = 0
		t.root = sha256.Sum256(nil)
		return nil
	}

	for i, d := range digests {
		leaf := [1 + sha256.Size]byte{LeafPrefix}
		copy(leaf[1:], d[:])
		t.levels[0][i] = sha256.Sum256(leaf[:])
	}

	l := 0
	w := len(digests)

	for w > 1 {
		b := [1 + 2*sha256.Size]byte{NodePrefix}

		wn := 0

		for i := 0; i+1 < w; i += 2 {
			copy(b[1:], t.levels[l][i][:])
			copy(b[1+sha256.Size:], t.levels[l][i+1][:])
			t.levels[l+1][wn] = sha256.Sum256(b[:])
			wn++
		}

		if w%2 == 1 {
			t.levels[l+1][wn] = t.levels[l][w-1]
			wn++
		}

		l++
		w = wn
	}

	t.width = len(digests)
	t.root = t.levels[l][0]

	return nil
}

func (t *HTree) Root() [sha256.Size]byte {
	return t.root
}

// InclusionProof returns the shortest list of additional nodes required to compute the root
// It's an adaption from the algorithm for proof construction at github.com/codenotary/merkletree
func (t *HTree) InclusionProof(i int) (proof *InclusionProof, err error) {
	if i >= t.width {
		return nil, ErrIllegalArguments
	}

	m := i
	n := t.width

	var offset int
	var l int
	var r int

	proof = &InclusionProof{
		Leaf:  i,
		Width: t.width,
	}

	if t.width == 1 {
		return
	}

	for {
		d := bits.Len(uint(n - 1))
		k := 1 << (d - 1)
		if m < k {
			l, r = offset+k, offset+n-1
			n = k
		} else {
			l, r = offset, offset+k-1
			m = m - k
			n = n - k
			offset += k
		}

		layer := bits.Len(uint(r - l))
		index := l / (1 << layer)

		proof.Terms = append([][sha256.Size]byte{t.levels[layer][index]}, proof.Terms...)

		if n < 1 || (n == 1 && m == 0) {
			return
		}
	}
}

func VerifyInclusion(proof *InclusionProof, digest, root [sha256.Size]byte) bool {
	if proof == nil {
		return false
	}

	leaf := [1 + sha256.Size]byte{LeafPrefix}
	copy(leaf[1:], digest[:])

	calcRoot := sha256.Sum256(leaf[:])
	i := proof.Leaf
	r := proof.Width - 1

	for _, t := range proof.Terms {
		b := [1 + 2*sha256.Size]byte{NodePrefix}

		if i%2 == 0 && i != r {
			copy(b[1:], calcRoot[:])
			copy(b[1+sha256.Size:], t[:])
		} else {
			copy(b[1:], t[:])
			copy(b[1+sha256.Size:], calcRoot[:])
		}

		calcRoot = sha256.Sum256(b[:])
		i /= 2
		r /= 2
	}

	return i == r && root == calcRoot
}

// Defensive budgets used when validating a multi proof. A verifier has no
// HTree instance to bound sizes, so untrusted widths and term counts are
// rejected before any allocation takes place.
const (
	maxMultiProofLeaves = 1 << 30
	maxMultiProofTerms  = 1 << 30
	maxMultiProofHeight = 31
)

// MultiProofTerm is one auxiliary subtree of a MultiInclusionProof. It carries
// the digest of the node stored at (Layer, Index), exactly as computed by
// BuildWith. The node covers the leaf range
// [Index*2^Layer, min(Width, (Index+1)*2^Layer)), so nodes produced by the
// odd-tail promotion rule are addressed the same way as any other node.
type MultiProofTerm struct {
	Layer int
	Index int
	Hash  [sha256.Size]byte
}

// MultiInclusionProof proves the inclusion of several leaf digests in the tree
// with the given Width. Leaves holds the canonical (strictly increasing) leaf
// positions and Terms holds the minimal set of auxiliary subtrees required to
// recompute the root. Terms are ordered by the first leaf index they cover,
// which is the canonical order produced by MultiInclusionProof.
//
// The structure owns all its memory: rebuilding the originating HTree never
// mutates an already issued proof.
type MultiInclusionProof struct {
	Width  int
	Leaves []int
	Terms  []MultiProofTerm
}

func hashLeaf(digest [sha256.Size]byte) [sha256.Size]byte {
	b := [1 + sha256.Size]byte{LeafPrefix}
	copy(b[1:], digest[:])
	return sha256.Sum256(b[:])
}

func hashNode(left, right [sha256.Size]byte) [sha256.Size]byte {
	b := [1 + 2*sha256.Size]byte{NodePrefix}
	copy(b[1:], left[:])
	copy(b[1+sha256.Size:], right[:])
	return sha256.Sum256(b[:])
}

// layerNodeCount returns the number of valid nodes stored at the given layer
// for a tree of the given width.
func layerNodeCount(width, layer int) int {
	return ((width - 1) >> layer) + 1
}

// termCoverage returns the half-open leaf range covered by node (layer, index).
func termCoverage(width, layer, index int) (start, end int) {
	span := 1 << layer
	start = index * span
	end = start + span
	if end > width {
		end = width
	}
	return start, end
}

// subtreeCoordinates returns the (layer, index) coordinate of the stored node
// rooting the subtree spanning size leaves starting at offset. It is the same
// coordinate rule used by InclusionProof.
func subtreeCoordinates(offset, size int) (layer, index int) {
	layer = bits.Len(uint(size - 1))
	index = offset >> layer
	return layer, index
}

// MultiInclusionProof builds a multi-leaf inclusion proof for the given leaf
// positions. Auxiliary subtrees shared between several leaf paths are emitted
// only once, so the result is the minimal set of nodes required to recompute
// the root. It is a generalization of the single-leaf InclusionProof algorithm
// and does not perform any I/O.
//
// Leaves must be non-empty, within [0, tree width), strictly increasing and
// free of duplicates. A zero-leaf request is explicitly rejected.
func (t *HTree) MultiInclusionProof(leaves []int) (*MultiInclusionProof, error) {
	if t.width <= 0 || len(leaves) == 0 || len(leaves) > maxMultiProofLeaves {
		return nil, ErrIllegalArguments
	}

	prev := -1

	for _, leaf := range leaves {
		if leaf < 0 || leaf >= t.width || leaf <= prev {
			return nil, ErrIllegalArguments
		}

		prev = leaf
	}

	proof := &MultiInclusionProof{
		Width:  t.width,
		Leaves: append([]int(nil), leaves...),
		Terms:  make([]MultiProofTerm, 0, t.width),
	}

	proof.Terms = t.appendMultiTerms(0, t.width, 0, len(leaves), leaves, proof.Terms)

	if len(proof.Terms) > maxMultiProofTerms {
		return nil, ErrMaxWidthExceeded
	}

	return proof, nil
}

// appendMultiTerms emits the auxiliary subtrees needed by the targets
// leaves[lo:hi], all of which live within the subtree of size nodes starting
// at leaf offset. A subtree half containing no target leaf is emitted as a
// single term instead of descending into it, which is what makes shared path
// nodes appear at most once.
func (t *HTree) appendMultiTerms(
	offset, size, lo, hi int,
	leaves []int,
	terms []MultiProofTerm,
) []MultiProofTerm {
	if size == 1 {
		return terms
	}

	k := 1 << (bits.Len(uint(size-1)) - 1)
	split := offset + k
	mid := lo + sort.SearchInts(leaves[lo:hi], split)

	if mid == lo {
		terms = append(terms, t.auxTerm(offset, k))
	} else {
		terms = t.appendMultiTerms(offset, k, lo, mid, leaves, terms)
	}

	if mid == hi {
		terms = append(terms, t.auxTerm(split, size-k))
	} else {
		terms = t.appendMultiTerms(split, size-k, mid, hi, leaves, terms)
	}

	return terms
}

func (t *HTree) auxTerm(offset, size int) MultiProofTerm {
	layer, index := subtreeCoordinates(offset, size)
	return MultiProofTerm{
		Layer: layer,
		Index: index,
		Hash:  t.levels[layer][index],
	}
}

// VerifyMultiInclusion verifies a multi-leaf inclusion proof against a trusted
// root. It is a pure function: no HTree instance, storage access or undisclosed
// leaf content is required. Besides the proof and the trusted root, it only
// consumes the expected width and leaf positions, and the corresponding leaf
// digests, which must be aligned position-by-position with leaves.
//
// The proof is rejected unless its declared width and leaf set exactly match
// the expected ones, every term is in range, terms are canonically ordered and
// cover disjoint subtrees that never overlap a requested leaf, all terms are
// strictly necessary and consumed exactly once, and the recomputed root equals
// the trusted one. This prevents proving a different statement by choosing
// arbitrary leaf positions or widths.
func VerifyMultiInclusion(
	proof *MultiInclusionProof,
	expectedWidth int,
	leaves []int,
	digests [][sha256.Size]byte,
	root [sha256.Size]byte,
) bool {
	if proof == nil || len(leaves) == 0 || len(leaves) != len(digests) {
		return false
	}

	width, ok := validProofWidth(expectedWidth)
	if !ok || proof.Width != width {
		return false
	}

	if !validSortedLeaves(leaves, width) || !sameLeafSet(proof.Leaves, leaves) {
		return false
	}

	termsByLayer, ok := validateMultiTerms(proof.Terms, width, leaves)
	if !ok {
		return false
	}

	calcRoot, ok := reconstructMultiRoot(width, leaves, digests, proof.Terms, termsByLayer)

	return ok && calcRoot == root
}

func validProofWidth(width int) (int, bool) {
	if width <= 0 || width > maxMultiProofLeaves {
		return 0, false
	}

	if bits.Len(uint(width-1))+1 > maxMultiProofHeight {
		return 0, false
	}

	return width, true
}

func validSortedLeaves(leaves []int, width int) bool {
	if len(leaves) == 0 || len(leaves) > width || len(leaves) > maxMultiProofLeaves {
		return false
	}

	prev := -1

	for _, leaf := range leaves {
		if leaf < 0 || leaf >= width || leaf <= prev {
			return false
		}

		prev = leaf
	}

	return true
}

func sameLeafSet(proofLeaves, expectedLeaves []int) bool {
	if len(proofLeaves) != len(expectedLeaves) {
		return false
	}

	for i := range proofLeaves {
		if proofLeaves[i] != expectedLeaves[i] {
			return false
		}
	}

	return true
}

// validateMultiTerms enforces coordinate bounds, canonical ordering, disjoint
// term coverage and separation from the requested leaf set. It also groups
// term positions by layer for the reconstruction pass.
func validateMultiTerms(
	terms []MultiProofTerm,
	width int,
	leaves []int,
) ([][]int, bool) {
	if len(terms) > maxMultiProofTerms || len(terms) >= width {
		return nil, false
	}

	topLayer := bits.Len(uint(width - 1))
	termsByLayer := make([][]int, topLayer+1)

	prevStart := -1
	prevEnd := 0
	leafCursor := 0

	for pos, term := range terms {
		if term.Layer < 0 || term.Layer > topLayer {
			return nil, false
		}

		if term.Index < 0 || term.Index >= layerNodeCount(width, term.Layer) {
			return nil, false
		}

		start, end := termCoverage(width, term.Layer, term.Index)

		// Canonical order and disjoint coverage.
		if start <= prevStart || start < prevEnd {
			return nil, false
		}

		// No term may cover a requested leaf.
		for leafCursor < len(leaves) && leaves[leafCursor] < start {
			leafCursor++
		}

		if leafCursor < len(leaves) && leaves[leafCursor] < end {
			return nil, false
		}

		termsByLayer[term.Layer] = append(termsByLayer[term.Layer], pos)
		prevStart = start
		prevEnd = end
	}

	return termsByLayer, true
}

// reconstructMultiRoot replays BuildWith using only the requested leaf
// digests and the auxiliary terms. Term-covered subtrees stay opaque until the
// layer of their term is reached, then the supplied digest is injected exactly
// once. The walk mirrors the pair hashing and odd-tail promotion rules of
// BuildWith, including the single leaf-prefix hashing of each digest.
func reconstructMultiRoot(
	width int,
	leaves []int,
	digests [][sha256.Size]byte,
	terms []MultiProofTerm,
	termsByLayer [][]int,
) ([sha256.Size]byte, bool) {
	known := make(map[int][sha256.Size]byte, len(leaves))
	covered := make(map[int]int, len(terms))

	for pos, term := range terms {
		start, end := termCoverage(width, term.Layer, term.Index)

		for idx := start; idx < end; idx++ {
			if _, exists := covered[idx]; exists {
				return [sha256.Size]byte{}, false
			}

			covered[idx] = pos
		}
	}

	for pos, leaf := range leaves {
		if _, exists := covered[leaf]; exists {
			return [sha256.Size]byte{}, false
		}

		known[leaf] = hashLeaf(digests[pos])
	}

	w := width

	for layer := 0; w > 1; layer++ {
		for _, pos := range termsByLayer[layer] {
			term := terms[pos]

			if _, exists := known[term.Index]; exists {
				return [sha256.Size]byte{}, false
			}

			owner, exists := covered[term.Index]
			if !exists || owner != pos {
				return [sha256.Size]byte{}, false
			}

			known[term.Index] = term.Hash
			delete(covered, term.Index)
		}

		nextKnown := make(map[int][sha256.Size]byte)
		nextCovered := make(map[int]int)

		if !mergeMultiLayer(w, known, covered, nextKnown, nextCovered) {
			return [sha256.Size]byte{}, false
		}

		known = nextKnown
		covered = nextCovered
		w = (w + 1) >> 1
	}

	if len(covered) != 0 || len(known) != 1 {
		return [sha256.Size]byte{}, false
	}

	calcRoot, exists := known[0]

	return calcRoot, exists
}

func mergeMultiLayer(
	w int,
	known map[int][sha256.Size]byte,
	covered map[int]int,
	nextKnown map[int][sha256.Size]byte,
	nextCovered map[int]int,
) bool {
	idx := 0

	for ; idx+1 < w; idx += 2 {
		leftHash, leftKnown := known[idx]
		rightHash, rightKnown := known[idx+1]

		if leftKnown && rightKnown {
			nextKnown[idx>>1] = hashNode(leftHash, rightHash)
			continue
		}

		leftOwner, leftCovered := covered[idx]
		rightOwner, rightCovered := covered[idx+1]

		if !leftCovered || !rightCovered || leftOwner != rightOwner {
			return false
		}

		nextCovered[idx>>1] = leftOwner
	}

	if w%2 == 1 {
		if hash, exists := known[idx]; exists {
			nextKnown[idx>>1] = hash
			return true
		}

		owner, exists := covered[idx]
		if !exists {
			return false
		}

		nextCovered[idx>>1] = owner
	}

	return true
}
