//  Tool to extract unigram counts
//
//  GloVe: Global Vectors for Word Representation
//  Copyright (c) 2014 The Board of Trustees of
//  The Leland Stanford Junior University. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//
//  For more information, bug reports, fixes, contact:
//    Jeffrey Pennington (jpennin@stanford.edu)
//    Christopher Manning (manning@cs.stanford.edu)
//    https://github.com/stanfordnlp/GloVe/
//    GlobalVectors@googlegroups.com
//    http://nlp.stanford.edu/projects/glove/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_STRING_LENGTH 1000
#define TSIZE   1048576
#define SEED    1159241

#define HASHFN  bitwisehash

typedef struct vocabulary {
    char *word;
    long long count;
} VOCAB;

//각 해시값에 해당되는 배열 head에 linked list로 같은 해시값을 같는 단어들이 연결되어있는 테이블 
typedef struct hashrec {
    char *word;
    long long count;
    struct hashrec *next;
} HASHREC;

int verbose = 2; // 0, 1, or 2
long long min_count = 1; // min occurrences for inclusion in vocab
long long max_vocab = 0; // max_vocab = 0 for no limit


/* Efficient string comparison */
int scmp( char *s1, char *s2 ) {
    while (*s1 != '\0' && *s1 == *s2) {s1++; s2++;}
    return *s1 - *s2;
}
//같은 단어면 0반환
//아니면 단어의 차이값 

/* Vocab frequency comparison; break ties alphabetically */
int CompareVocabTie(const void *a, const void *b) { //호춝 코드(219)
    long long c;
    if ( (c = ((VOCAB *) b)->count - ((VOCAB *) a)->count) != 0) return ( c > 0 ? 1 : -1 ); //b가 a보다 더 크면 1, 아니면 -1 반환
    else return (scmp(((VOCAB *) a)->word,((VOCAB *) b)->word)); //a가 더 먼저 나오면 음의 값을, b가 더 먼저 나오면 양의 값을, 같으면 0 
    //사전순으로 더 앞에 나오는 단어가 값이 작음
    
}

/* Vocab frequency comparison; no tie-breaker */
int CompareVocab(const void *a, const void *b) { //호출 코드(217)
    long long c;
    if ( (c = ((VOCAB *) b)->count - ((VOCAB *) a)->count) != 0) return ( c > 0 ? 1 : -1 ); //b가 a보다 크면 1, 아니면 -1 반환
    else return 0; //같으면 0 반환
}

/* Move-to-front hashing and hash function from Hugh Williams, http://www.seg.rmit.edu.au/code/zwh-ipl/ */

/* Simple bitwise hash function */
unsigned int bitwisehash(char *word, int tsize, unsigned int seed) {
    char c;
    unsigned int h;
    h = seed;
    for ( ; (c = *word) != '\0'; word++) h ^= ((h << 5) + c + (h >> 2));
    return (unsigned int)((h & 0x7fffffff) % tsize);
}

/* Create hash table, initialise pointers to NULL */
HASHREC ** inithashtable() {
    int i;
    HASHREC **ht; //해시테이블 포인터 생성
    ht = (HASHREC **) malloc( sizeof(HASHREC *) * TSIZE ); //tsize 길이의 hashtable 생성해서 포인터가 가리킴
    for (i = 0; i < TSIZE; i++) ht[i] = (HASHREC *) NULL; //모든 요소를 null로 초기화
    return ht; //생성된 해시테이블의 포인터를 반환
}

/* Search hash table for given string, insert if not found */
void hashinsert(HASHREC **ht, char *w) { //호출 코드(192)
    HASHREC     *htmp, *hprv; //htmp: 현재 노드, hprv: 이전 노드
    unsigned int hval = HASHFN(w, TSIZE, SEED); // 해당 단어의 해시값 생성(bitwisehash 사용)
    
    //해시값에 해당되는 노드부터 "현재 노드가 null이거나, node에 저장된 word와 저장할 word가 같을 때"까지 순차탐색
    //해시값이 같은 단어 노드들을 탐색하는 것
    for (hprv = NULL, htmp = ht[hval]; htmp != NULL && scmp(htmp->word, w) != 0; hprv = htmp, htmp = htmp->next);
    if (htmp == NULL) { //현재 노드가 비어있다면
        htmp = (HASHREC *) malloc( sizeof(HASHREC) ); //노드 할당
        htmp->word = (char *) malloc( strlen(w) + 1 ); //word를 저장할 공간 할당
        strcpy(htmp->word, w); //word를 복사(w는 포인터이므로 저장하지 않고 복사)
        htmp->count = 1; //원래 null이었으므로 현재  빈도수는 1
        htmp->next = NULL; 
        if ( hprv==NULL ) //이전 노드가 null이면
            ht[hval] = htmp; //첫 노드가 지금 노드
        else
            hprv->next = htmp; //이전 노드 다음에 연결
    }
    else { //이미 테이블에 저장된 단어라면
        /* new records are not moved to front */
        htmp->count++; //빈도수를 증가
        if (hprv != NULL) { //이전 노드가 존재했다면 <처음>-...-<이전>-<현재>-<다음> => <현재>-<처음>-...-<이전>-<다음> 
            /* move to front on access */
            hprv->next = htmp->next; //이전 노드에 다음 노드를 연결 <이전>-<다음>
            htmp->next = ht[hval]; //현재 노드 뒤에 처음 노드를 연결
            ht[hval] = htmp; //최종적으로 가장 앞에 있는 노드는 현재 노드
        }
    }
    return;
}

/* Read word from input stream. Return 1 when encounter '\n' or EOF (but separate from word), 0 otherwise.
   Words can be separated by space(s), tab(s), or newline(s). Carriage return characters are just ignored.
   (Okay for Windows, but not for Mac OS 9-. Ignored even if by themselves or in words.)
   A newline is taken as indicating a new document (contexts won't cross newline).
   Argument word array is assumed to be of size MAX_STRING_LENGTH.
   words will be truncated if too long. They are truncated with some care so that they
   cannot truncate in the middle of a utf-8 character, but
   still little to no harm will be done for other encodings like iso-8859-1.
   (This function appears identically copied in vocab_count.c and cooccur.c.)
 */
int get_word(char *word, FILE *fin) { //호출 코드(186)
    int i = 0, ch;
    for ( ; ; ) {
        ch = fgetc(fin); //character 단위로 읽음
        if (ch == '\r') continue;
        if (i == 0 && ((ch == '\n') || (ch == EOF))) {
            word[i] = 0;
            return 1; //처음부터 '\n' 또는 파일의 끝, 1반환
        }
        if (i == 0 && ((ch == ' ') || (ch == '\t'))) continue; // skip leading space //처음부터 공백문자는 skip
        if ((ch == EOF) || (ch == ' ') || (ch == '\t') || (ch == '\n')) { //문자열을 입력받은 뒤 공백, 줄바꿈 문자 들어옴
            if (ch == '\n') ungetc(ch, fin); // return the newline next time as document ender //ch를 다시 stream에 집어넣음
            break; //for문 종료
        }
        if (i < MAX_STRING_LENGTH - 1) //하나의 단어가 string max length보다 커지면 안됨
          word[i++] = ch; // don't allow words to exceed MAX_STRING_LENGTH //word에 ch를 넣고 i를 1 증가시킴
    }
    word[i] = 0; //null terminate
    // avoid truncation destroying a multibyte UTF-8 char except if only thing on line (so the i > x tests won't overwrite word[0])
    // see https://en.wikipedia.org/wiki/UTF-8#Description
    if (i == MAX_STRING_LENGTH - 1 && (word[i-1] & 0x80) == 0x80) {
        if ((word[i-1] & 0xC0) == 0xC0) {
            word[i-1] = '\0';
        } else if (i > 2 && (word[i-2] & 0xE0) == 0xE0) {
            word[i-2] = '\0';
        } else if (i > 3 && (word[i-3] & 0xF8) == 0xF0) {
            word[i-3] = '\0';
        }
    }
    return 0;
}

int get_counts() { //main에서 
    long long i = 0, j = 0, vocab_size = 12500;
    // char format[20];
    char str[MAX_STRING_LENGTH + 1];
    HASHREC **vocab_hash = inithashtable(); //초기화된 해시테이블 
    HASHREC *htmp; 
    VOCAB *vocab; //vocab 배열 생성
    FILE *fid = stdin; //input 파일 (CORPUS="text8")
    
    fprintf(stderr, "BUILDING VOCABULARY\n");
    if (verbose > 1) fprintf(stderr, "Processed %lld tokens.", i); //verbose == 2이면
    // sprintf(format,"%%%ds",MAX_STRING_LENGTH);
    while ( ! feof(fid)) {
        // Insert all tokens into hashtable
        int nl = get_word(str, fid); //word 하나를 읽었으면 0, 못 읽었으면 1 / str에 한 단어를 
        if (nl) continue; // just a newline marker or feof 
        if (strcmp(str, "<unk>") == 0) { //<unk>가 corpus에 있으면 안된다는 뜻
            fprintf(stderr, "\nError, <unk> vector found in corpus.\nPlease remove <unk>s from your corpus (e.g. cat text8 | sed -e 's/<unk>/<raw_unk>/g' > text8.new)");
            return 1;
        }
        hashinsert(vocab_hash, str); //해시테이블에 단어 저장
        if (((++i)%100000) == 0) if (verbose > 1) fprintf(stderr,"\033[11G%lld tokens.", i);
    }
    if (verbose > 1) fprintf(stderr, "\033[0GProcessed %lld tokens.\n", i);
    //input file의 모든 단어를 hashtable에 기록함
   
    vocab = malloc(sizeof(VOCAB) * vocab_size); 
    for (i = 0; i < TSIZE; i++) { // Migrate vocab to array
        htmp = vocab_hash[i]; 
        while (htmp != NULL) {
            vocab[j].word = htmp->word;
            vocab[j].count = htmp->count;
            j++;
            if (j>=vocab_size) {
                vocab_size += 2500; //vocab_size를 유동적으로 늘려줌
                vocab = (VOCAB *)realloc(vocab, sizeof(VOCAB) * vocab_size); //realloc: 할당된 동적메모리 크기를 수정
            }
            htmp = htmp->next;
        }
    }
    if (verbose > 1) fprintf(stderr, "Counted %lld unique words.\n", j); //서로 다른 단어의 수는 j
    //현재 vocabulary의 크기가 주어진 max보다 크다면
    if (max_vocab > 0 && max_vocab < j) 
        // If the vocabulary exceeds limit, first sort full vocab by frequency without alphabetical tie-breaks.
        // This results in pseudo-random ordering for words with same frequency, so that when truncated, the words span whole alphabet
        qsort(vocab, j, sizeof(VOCAB), CompareVocab); //빈도 순으로 내림차순 정렬
    else max_vocab = j; //기존 max보다 j가 작으면 max값을 j로 설정
    qsort(vocab, max_vocab, sizeof(VOCAB), CompareVocabTie); //After (possibly) truncating, sort (possibly again), breaking ties alphabetically
    //빈도 수가 적은 단어들은 잘라버림(max를 넘는 부분부터 사용 안함)
    
    /*stdlib.h의 qsort
    void qsort(void *base, size_t num, size_t width, int(*compare)(const void*, const void*));
    base: 정렬할 배열의 포인터
    num: 배열 요소의 수
    width: 배열 요소의 크기
    (*compare): 비교 함수의 포인터
        빈도수 a가 b보다 크면 -> -1반환, a가 b보다 작으면 -> 1반환, a와 b가 같으면 -> 0반환 or 단어 사전 순서로 정렬
        => 반환값이 1인 경우 swap: 내림차순으로 정렬
    */

    for (i = 0; i < max_vocab; i++) {
        if (vocab[i].count < min_count) { // If a minimum frequency cutoff exists, truncate vocabulary
            if (verbose > 0) fprintf(stderr, "Truncating vocabulary at min count %lld.\n",min_count);
            break;
        } //사용자가 설정한 min_count보다 빈도수가 작은 단어들은 고려하지 않음.
        printf("%s %lld\n",vocab[i].word,vocab[i].count); //=>max_count보다 빈도수가 큰 단어-빈도수를 output file에 저장
    }
    
    if (i == max_vocab && max_vocab < j) if (verbose > 0) fprintf(stderr, "Truncating vocabulary at size %lld.\n", max_vocab);
    fprintf(stderr, "Using vocabulary of size %lld.\n\n", i);
    return 0;
}

int find_arg(char *str, int argc, char **argv) {//main에서 호출, 입력 인자를 확인함
    int i;
    for (i = 1; i < argc; i++) {
        if (!scmp(str, argv[i])) {
            if (i == argc - 1) {
                printf("No argument given for %s\n", str);
                exit(1);
            }
            return i;
        }
    }
    return -1;
}

int main(int argc, char **argv) {
    int i;
    if (argc == 1) {
        printf("Simple tool to extract unigram counts\n");
        printf("Author: Jeffrey Pennington (jpennin@stanford.edu)\n\n");
        printf("Usage options:\n");
        printf("\t-verbose <int>\n");
        printf("\t\tSet verbosity: 0, 1, or 2 (default)\n");
        printf("\t-max-vocab <int>\n");
        printf("\t\tUpper bound on vocabulary size, i.e. keep the <int> most frequent words. The minimum frequency words are randomly sampled so as to obtain an even distribution over the alphabet.\n");
        printf("\t-min-count <int>\n");
        printf("\t\tLower limit such that words which occur fewer than <int> times are discarded.\n");
        printf("\nExample usage:\n");
        printf("./vocab_count -verbose 2 -max-vocab 100000 -min-count 10 < corpus.txt > vocab.txt\n");
        return 0;
    }
    
    if ((i = find_arg((char *)"-verbose", argc, argv)) > 0) verbose = atoi(argv[i + 1]); //verbose parameter를 받았으면 해당 값을 저장
    if ((i = find_arg((char *)"-max-vocab", argc, argv)) > 0) max_vocab = atoll(argv[i + 1]); //같은
    if ((i = find_arg((char *)"-min-count", argc, argv)) > 0) min_count = atoll(argv[i + 1]); //같음
    return get_counts(); <--
}

