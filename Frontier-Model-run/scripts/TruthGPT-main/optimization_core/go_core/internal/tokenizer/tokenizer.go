// Package tokenizer provides fast BPE tokenization for TruthGPT.
package tokenizer

import (
	"bufio"
	"os"
	"regexp"
	"sort"
	"strings"
	"sync"
	"unicode/utf8"
)

// ════════════════════════════════════════════════════════════════════════════════
// TYPES
// ════════════════════════════════════════════════════════════════════════════════

// Token represents a single token.
type Token struct {
	ID   int
	Text string
}

// Encoding represents the result of tokenization.
type Encoding struct {
	IDs      []int
	Tokens   []string
	Offsets  [][2]int // (start, end) byte offsets
	NumWords int
}

// TokenizerConfig holds configuration for the tokenizer.
type TokenizerConfig struct {
	VocabFile       string
	MergesFile      string
	MaxLength       int
	Padding         bool
	Truncation      bool
	PadToken        string
	UnkToken        string
	BosToken        string
	EosToken        string
	AddSpecialTokens bool
}

// DefaultConfig returns default tokenizer configuration.
func DefaultConfig() TokenizerConfig {
	return TokenizerConfig{
		MaxLength:        512,
		Padding:          false,
		Truncation:       true,
		PadToken:         "<pad>",
		UnkToken:         "<unk>",
		BosToken:         "<s>",
		EosToken:         "</s>",
		AddSpecialTokens: true,
	}
}

// ════════════════════════════════════════════════════════════════════════════════
// BPE TOKENIZER
// ════════════════════════════════════════════════════════════════════════════════

// BPETokenizer implements Byte-Pair Encoding tokenization.
type BPETokenizer struct {
	mu       sync.RWMutex
	config   TokenizerConfig
	vocab    map[string]int
	reverseVocab map[int]string
	merges   map[[2]string]int
	pattern  *regexp.Regexp
	cache    map[string][]string
	cacheMu  sync.RWMutex
}

// NewBPETokenizer creates a new BPE tokenizer.
func NewBPETokenizer(config TokenizerConfig) *BPETokenizer {
	t := &BPETokenizer{
		config:       config,
		vocab:        make(map[string]int),
		reverseVocab: make(map[int]string),
		merges:       make(map[[2]string]int),
		cache:        make(map[string][]string),
		pattern:      regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`),
	}
	return t
}

// LoadVocab loads vocabulary from file.
func (t *BPETokenizer) LoadVocab(path string) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	id := 0
	for scanner.Scan() {
		token := strings.TrimSpace(scanner.Text())
		if token == "" {
			continue
		}
		t.vocab[token] = id
		t.reverseVocab[id] = token
		id++
	}
	return scanner.Err()
}

// LoadMerges loads BPE merges from file.
func (t *BPETokenizer) LoadMerges(path string) error {
	t.mu.Lock()
	defer t.mu.Unlock()

	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	rank := 0
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.Split(line, " ")
		if len(parts) != 2 {
			continue
		}
		t.merges[[2]string{parts[0], parts[1]}] = rank
		rank++
	}
	return scanner.Err()
}

// Encode tokenizes text into token IDs.
func (t *BPETokenizer) Encode(text string) *Encoding {
	t.mu.RLock()
	defer t.mu.RUnlock()

	// Find all matches
	matches := t.pattern.FindAllStringIndex(text, -1)
	words := make([]string, 0, len(matches))
	offsets := make([][2]int, 0, len(matches))

	for _, match := range matches {
		words = append(words, text[match[0]:match[1]])
		offsets = append(offsets, [2]int{match[0], match[1]})
	}

	ids := make([]int, 0)
	tokens := make([]string, 0)
	finalOffsets := make([][2]int, 0)

	// Add BOS token if configured
	if t.config.AddSpecialTokens && t.config.BosToken != "" {
		if id, ok := t.vocab[t.config.BosToken]; ok {
			ids = append(ids, id)
			tokens = append(tokens, t.config.BosToken)
			finalOffsets = append(finalOffsets, [2]int{0, 0})
		}
	}

	// Tokenize each word
	for i, word := range words {
		wordTokens := t.bpe(word)
		for _, tok := range wordTokens {
			if id, ok := t.vocab[tok]; ok {
				ids = append(ids, id)
				tokens = append(tokens, tok)
				finalOffsets = append(finalOffsets, offsets[i])
			} else if id, ok := t.vocab[t.config.UnkToken]; ok {
				ids = append(ids, id)
				tokens = append(tokens, t.config.UnkToken)
				finalOffsets = append(finalOffsets, offsets[i])
			}
		}
	}

	// Add EOS token if configured
	if t.config.AddSpecialTokens && t.config.EosToken != "" {
		if id, ok := t.vocab[t.config.EosToken]; ok {
			ids = append(ids, id)
			tokens = append(tokens, t.config.EosToken)
			finalOffsets = append(finalOffsets, [2]int{len(text), len(text)})
		}
	}

	// Truncate if needed
	if t.config.Truncation && len(ids) > t.config.MaxLength {
		ids = ids[:t.config.MaxLength]
		tokens = tokens[:t.config.MaxLength]
		finalOffsets = finalOffsets[:t.config.MaxLength]
	}

	// Pad if needed
	if t.config.Padding && len(ids) < t.config.MaxLength {
		padID, _ := t.vocab[t.config.PadToken]
		for len(ids) < t.config.MaxLength {
			ids = append(ids, padID)
			tokens = append(tokens, t.config.PadToken)
			finalOffsets = append(finalOffsets, [2]int{0, 0})
		}
	}

	return &Encoding{
		IDs:      ids,
		Tokens:   tokens,
		Offsets:  finalOffsets,
		NumWords: len(words),
	}
}

// Decode converts token IDs back to text.
func (t *BPETokenizer) Decode(ids []int) string {
	t.mu.RLock()
	defer t.mu.RUnlock()

	var builder strings.Builder
	for _, id := range ids {
		if token, ok := t.reverseVocab[id]; ok {
			// Skip special tokens
			if token == t.config.PadToken || token == t.config.BosToken || token == t.config.EosToken {
				continue
			}
			builder.WriteString(token)
		}
	}
	return builder.String()
}

// bpe performs BPE on a single word.
func (t *BPETokenizer) bpe(word string) []string {
	// Check cache
	t.cacheMu.RLock()
	if cached, ok := t.cache[word]; ok {
		t.cacheMu.RUnlock()
		return cached
	}
	t.cacheMu.RUnlock()

	// Convert to characters
	chars := make([]string, 0, utf8.RuneCountInString(word))
	for _, r := range word {
		chars = append(chars, string(r))
	}

	if len(chars) == 0 {
		return nil
	}

	if len(chars) == 1 {
		return chars
	}

	// Apply BPE merges
	for {
		// Find best merge
		bestPair := [2]string{}
		bestRank := -1

		for i := 0; i < len(chars)-1; i++ {
			pair := [2]string{chars[i], chars[i+1]}
			if rank, ok := t.merges[pair]; ok {
				if bestRank == -1 || rank < bestRank {
					bestRank = rank
					bestPair = pair
				}
			}
		}

		if bestRank == -1 {
			break
		}

		// Apply merge
		newChars := make([]string, 0, len(chars))
		i := 0
		for i < len(chars) {
			if i < len(chars)-1 && chars[i] == bestPair[0] && chars[i+1] == bestPair[1] {
				newChars = append(newChars, chars[i]+chars[i+1])
				i += 2
			} else {
				newChars = append(newChars, chars[i])
				i++
			}
		}
		chars = newChars
	}

	// Store in cache
	t.cacheMu.Lock()
	t.cache[word] = chars
	t.cacheMu.Unlock()

	return chars
}

// VocabSize returns the vocabulary size.
func (t *BPETokenizer) VocabSize() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return len(t.vocab)
}

// GetID returns the ID for a token.
func (t *BPETokenizer) GetID(token string) (int, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	id, ok := t.vocab[token]
	return id, ok
}

// GetToken returns the token for an ID.
func (t *BPETokenizer) GetToken(id int) (string, bool) {
	t.mu.RLock()
	defer t.mu.RUnlock()
	token, ok := t.reverseVocab[id]
	return token, ok
}

// ════════════════════════════════════════════════════════════════════════════════
// BATCH TOKENIZER
// ════════════════════════════════════════════════════════════════════════════════

// BatchTokenizer provides parallel tokenization.
type BatchTokenizer struct {
	tokenizer *BPETokenizer
	workers   int
}

// NewBatchTokenizer creates a new batch tokenizer.
func NewBatchTokenizer(tokenizer *BPETokenizer, workers int) *BatchTokenizer {
	if workers <= 0 {
		workers = 4
	}
	return &BatchTokenizer{
		tokenizer: tokenizer,
		workers:   workers,
	}
}

// EncodeBatch tokenizes multiple texts in parallel.
func (b *BatchTokenizer) EncodeBatch(texts []string) []*Encoding {
	results := make([]*Encoding, len(texts))
	var wg sync.WaitGroup

	// Create work channel
	work := make(chan int, len(texts))
	for i := range texts {
		work <- i
	}
	close(work)

	// Start workers
	for w := 0; w < b.workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range work {
				results[i] = b.tokenizer.Encode(texts[i])
			}
		}()
	}

	wg.Wait()
	return results
}

// PadBatch pads a batch of encodings to the same length.
func (b *BatchTokenizer) PadBatch(encodings []*Encoding) ([][]int, [][]int) {
	if len(encodings) == 0 {
		return nil, nil
	}

	// Find max length
	maxLen := 0
	for _, enc := range encodings {
		if len(enc.IDs) > maxLen {
			maxLen = enc.IDs
		}
	}

	padID, _ := b.tokenizer.GetID(b.tokenizer.config.PadToken)

	// Pad all sequences
	inputIDs := make([][]int, len(encodings))
	attentionMask := make([][]int, len(encodings))

	for i, enc := range encodings {
		inputIDs[i] = make([]int, maxLen)
		attentionMask[i] = make([]int, maxLen)

		copy(inputIDs[i], enc.IDs)
		for j := 0; j < len(enc.IDs); j++ {
			attentionMask[i][j] = 1
		}
		for j := len(enc.IDs); j < maxLen; j++ {
			inputIDs[i][j] = padID
			attentionMask[i][j] = 0
		}
	}

	return inputIDs, attentionMask
}

// ════════════════════════════════════════════════════════════════════════════════
// VOCABULARY BUILDER
// ════════════════════════════════════════════════════════════════════════════════

// VocabBuilder builds BPE vocabulary from corpus.
type VocabBuilder struct {
	minFreq   int
	vocabSize int
}

// NewVocabBuilder creates a new vocabulary builder.
func NewVocabBuilder(vocabSize, minFreq int) *VocabBuilder {
	return &VocabBuilder{
		vocabSize: vocabSize,
		minFreq:   minFreq,
	}
}

// Build builds vocabulary and merges from texts.
func (v *VocabBuilder) Build(texts []string) (map[string]int, [][2]string) {
	// Count word frequencies
	wordFreq := make(map[string]int)
	pattern := regexp.MustCompile(`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+`)

	for _, text := range texts {
		for _, match := range pattern.FindAllString(text, -1) {
			wordFreq[match]++
		}
	}

	// Initialize vocab with characters
	vocab := make(map[string]int)
	charFreq := make(map[string]int)

	for word, freq := range wordFreq {
		for _, r := range word {
			charFreq[string(r)] += freq
		}
	}

	id := 0
	for char, freq := range charFreq {
		if freq >= v.minFreq {
			vocab[char] = id
			id++
		}
	}

	// Build merges
	merges := make([][2]string, 0)

	for len(vocab) < v.vocabSize {
		// Count pairs
		pairFreq := make(map[[2]string]int)

		for word, freq := range wordFreq {
			chars := []string{}
			for _, r := range word {
				chars = append(chars, string(r))
			}

			for i := 0; i < len(chars)-1; i++ {
				pair := [2]string{chars[i], chars[i+1]}
				pairFreq[pair] += freq
			}
		}

		if len(pairFreq) == 0 {
			break
		}

		// Find best pair
		var bestPair [2]string
		bestFreq := 0
		for pair, freq := range pairFreq {
			if freq > bestFreq {
				bestFreq = freq
				bestPair = pair
			}
		}

		if bestFreq < v.minFreq {
			break
		}

		// Add to vocab
		merged := bestPair[0] + bestPair[1]
		vocab[merged] = id
		id++
		merges = append(merges, bestPair)

		// Update word frequencies
		newWordFreq := make(map[string]int)
		for word, freq := range wordFreq {
			newWord := strings.ReplaceAll(word, bestPair[0]+bestPair[1], merged)
			newWordFreq[newWord] = freq
		}
		wordFreq = newWordFreq
	}

	return vocab, merges
}

// ════════════════════════════════════════════════════════════════════════════════
// SPECIAL TOKENS
// ════════════════════════════════════════════════════════════════════════════════

// SpecialTokens defines standard special tokens.
type SpecialTokens struct {
	Pad   int
	Unk   int
	Bos   int
	Eos   int
	Mask  int
	Sep   int
	Cls   int
}

// GetSpecialTokens extracts special token IDs from tokenizer.
func GetSpecialTokens(t *BPETokenizer) SpecialTokens {
	st := SpecialTokens{
		Pad:  -1,
		Unk:  -1,
		Bos:  -1,
		Eos:  -1,
		Mask: -1,
		Sep:  -1,
		Cls:  -1,
	}

	if id, ok := t.GetID(t.config.PadToken); ok {
		st.Pad = id
	}
	if id, ok := t.GetID(t.config.UnkToken); ok {
		st.Unk = id
	}
	if id, ok := t.GetID(t.config.BosToken); ok {
		st.Bos = id
	}
	if id, ok := t.GetID(t.config.EosToken); ok {
		st.Eos = id
	}
	if id, ok := t.GetID("<mask>"); ok {
		st.Mask = id
	}
	if id, ok := t.GetID("<sep>"); ok {
		st.Sep = id
	}
	if id, ok := t.GetID("<cls>"); ok {
		st.Cls = id
	}

	return st
}












