import random

def hash(a, b, c, d):
	return 1000 * a + 100 * b + 10 * c + d

n = int(input())
was = {}
A = []
for _ in range(n):
	a, b, c, d, e, f = map(int, input().split())
	if b >= 3:
		b = random.randint(0, 2)
	if c >= 3:
		c = random.randint(0, 2)
	if d >= 2:
		d = random.randint(0, 1)
	h = hash(a, b, c, d)
	if h not in was:
		was[h] = True
		A.append([a, b, c, d])
print(len(A))
for i in range(len(A)):
	print(*A[i])

