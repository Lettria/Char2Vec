from .keyboards import *

"""Specific cases for accent"""
def compare_chars(c1, c2):
	if c1 in "aàâä" and c2 in "aàâä":
		return 0.5
	elif c1 in "çc" and c2 in "çc":
		return 0.5
	elif c1 in "eéèëê" and c2 in "eéèëê":
		return 0.5
	elif c1 in "oóòôö" and c2 in "oóòôö":
		return 0.5
	elif c1 in "uúùûü" and c2 in "uúùûü":
		return 0.5
	else:
		return 1

"""	Compute edit distance between two words(transpositions, deletions, insertions, replacement)."""
def levenshtein(word1, word2):
	matrix = []
	n = len(word1)
	m = len(word2)
	matrix.append([i for i in range(n + 1)])
	for i in range(1, (m + 1)):
		matrix.append([0] * (n + 1))
		matrix[i][0] = i
	for j in range(1, m + 1):
		for i in range(1, n + 1):
			if word1[i - 1] != word2[j - 1]:
				cost = compare_chars(word1[i - 1], word2[j - 1])
			else:
				cost = 0
			matrix[j][i] = min(matrix[j - 1][i] + 1,
								matrix[j][i - 1] + 1,
								matrix[j - 1][i - 1] + cost)
			if i >= 2 and j >= 2 and word1[i - 1] == word2[j - 2] and word1[i - 2] == word2[j - 1]:
				matrix[j][i] = min(matrix[j][i], matrix[j - 2][i - 2] + cost)
	return int(matrix[m][n])

"""	Compute relative edit distance based on azerty keyboard.
	Scale is not the same than 'normal' levenshtein"""
def levenshtein_relative(word1, word2):
	matrix = []
	n = len(word1)
	m = len(word2)
	matrix.append([i for i in range(n + 1)])
	for i in range(1, (m + 1)):
		matrix.append([0] * (n + 1))
		matrix[i][0] = i
	for j in range(1, m + 1):
		for i in range(1, n + 1):
			if word1[i - 1] != word2[j - 1]:
				cost = key_dist[word1[i - 1]][word2[j - 1]] # Min entre ca et touche d'apres?
			else:
				cost = 0
			matrix[j][i] = min(matrix[j - 1][i] + 1,
								matrix[j][i - 1] + 1,
								matrix[j - 1][i - 1] + cost)
			if i >= 2 and j >= 2 and word1[i - 1] == word2[j - 2] and word1[i - 2] == word2[j - 1]:
				matrix[j][i] = min(matrix[j][i], matrix[j - 2][i - 2] + cost)
	return int(matrix[m][n])
