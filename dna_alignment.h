#ifndef DNA_ALIGNMENT_H
#define DNA_ALIGNMENT_H

#include <stdint.h>
#include <stddef.h>

// DNA sequence alignment result structure
typedef struct {
    size_t num_matches;      // Total number of pattern matches found
    uint64_t checksum;       // Checksum for verification
    int* coverage;           // Coverage array (length = sequence_length)
} AlignmentResult;

// Algorithm types
typedef enum {
    ALG_BRUTE_FORCE,
    ALG_KMP,
    ALG_BOYER_MOORE,
    ALG_RABIN_KARP
} AlgorithmType;

/**
 * Perform DNA sequence alignment using specified algorithm
 * 
 * @param sequence: The DNA sequence to search in (null-terminated string)
 * @param sequence_len: Length of the sequence
 * @param patterns: Array of pattern strings to search for
 * @param num_patterns: Number of patterns
 * @param pattern_lengths: Array of lengths for each pattern
 * @param algorithm: Algorithm to use
 * @return AlignmentResult with matches, checksum, and coverage
 */
AlignmentResult* align_sequences(
    const char* sequence,
    size_t sequence_len,
    const char** patterns,
    size_t num_patterns,
    const size_t* pattern_lengths,
    AlgorithmType algorithm
);

/**
 * Free alignment result memory
 */
void free_alignment_result(AlignmentResult* result);

/**
 * Compare two alignment results for correctness verification
 * Returns 1 if results match, 0 otherwise
 */
int compare_results(const AlignmentResult* result1, const AlignmentResult* result2);

#endif // DNA_ALIGNMENT_H

