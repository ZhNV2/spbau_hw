def upd(dict, t):
	dict[t] = dict.get(t, 0) + 1

def check(dict, n):
	if len(dict) != n:
		print('ERROR!')

def unique(dict, t):
	return dict[t] == 1

def hash(a, b, c):
	return 100 * a + 10 * b + c

n = int(input())

dict1, dict2, dict3, dict4 = {}, {}, {}, {}
tests = []
for _ in range(n):
	a, b, c, d = map(int, input().split())
	t1 = hash(a, b, c)
	t2 = hash(a, b, d)
	t3 = hash(a, c, d)
	t4 = hash(b, c, d)
	upd(dict1, t1)
	upd(dict2, t2)
	upd(dict3, t3)
	upd(dict4, t4)
	tests.append([a, b, c, d])

check(dict1, 5 * 3 * 3)
check(dict2, 5 * 3 * 2)
check(dict3, 5 * 3 * 2)
check(dict4, 3 * 3 * 2)

print(n)
for test in tests:
	a, b, c, d = test[0], test[1], test[2], test[3]
	need = False
	need |= unique(dict1, hash(a, b, c))
	need |= unique(dict2, hash(a, b, d))
	need |= unique(dict3, hash(a, c, d))
	need |= unique(dict4, hash(b, c, d))
	print(chr(ord('A') + a), chr(ord('F') + b), chr(ord('K') + c), chr(ord('N') + d), end = '')
	if not need:
		print(' // can be skipped', end = '')
	print()


