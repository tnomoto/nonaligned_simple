#!/usr/bin/env python
# -*- coding: utf-8 -*- 

import sys
import codecs
import unicodedata
import math
import hashlib
import nltk
import argparse
import sentencepiece as sp

spm = sp.SentencePieceProcessor()

parser = argparse.ArgumentParser()

parser.add_argument('-a', '--a')
parser.add_argument('-n', '--n',type=int, default=10, metavar='MN',help='num of hypos')
parser.add_argument('-p', '--p',help="spm model")
parser.add_argument('-o', '--o')
parser.add_argument('-r', '--r', help="turk_test.ar")


args = parser.parse_args()


spm.Load(args.p)

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427

def u2a(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

n_hypos = args.n

def get_block(l, n):
	for i in range(0, len(l), n):
		yield list(map(lambda x:x.split('\t')[2], l[i:i+n]))

lines = [ l for l in codecs.open(args.a,'r',encoding='utf8') ]
l_ar = [ l.rstrip() for l in lines if l.startswith("S-")]
l_ke = [ l.rstrip() for l in lines if l.startswith("T-")]
l_hyp = [ l.rstrip() for l in lines if l.startswith("H-")]

hypo_l = []
for i in get_block(l_hyp,n_hypos):
	hypo_l.append(i)


l_ar =l_ar[:]
# l_ke =l_ke[:]
l_ke =l_ar[:]

hypo_l =hypo_l[:]


def spm_decode(str):
	out = spm.DecodePieces(str.split())
	return out.split()

best_sum = 0.0

sys_src= []
sys_hypos = []
fout = open (args.o,"w")

assert len(l_ar) == len(l_ke) == len(hypo_l)

for e in zip(l_ar, l_ke, hypo_l):
	source, reference, candidates = e
	src = [ x for x in spm_decode(source.split("\t")[1]) ]

	cand = [ x for x in spm_decode(candidates[0]) ]

	sys_src.append(' '.join(src))
	sys_hypos.append(' '.join(cand))
	fout.write("{}\n".format(' '.join(cand)))
