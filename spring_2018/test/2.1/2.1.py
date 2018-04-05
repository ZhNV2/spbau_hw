def hash(a, b):
	return 10 * a + b

def upd(dict, a, b):
	dict[hash(a, b)] = dict.get(hash(a, b), 0) + 1

def to_int(t):
	return 1 if t else 0

tests = []
for a in range(5):
	for b in range(3):
		for c in range(3):
			for d in range(2):
				tests.append([a, b, c, d])

res_tests = []
dict_a_b, dict_a_c, dict_a_d, dict_b_c, dict_b_d, dict_c_d = {}, {}, {}, {}, {}, {}
for test in tests:
	a, b, c, d = test[0], test[1], test[2], test[3]
	need = False
	need |= hash(a, b) not in dict_a_b 
	need |= hash(a, c) not in dict_a_c 
	need |= hash(a, d) not in dict_a_d 
	need |= hash(b, c) not in dict_b_c 
	need |= hash(b, d) not in dict_b_d 
	need |= hash(c, d) not in dict_c_d 
	if need:
		res_tests.append(test)
		upd(dict_a_b, a, b)
		upd(dict_a_c, a, c)
		upd(dict_a_d, a, d)
		upd(dict_b_c, b, c)
		upd(dict_b_d, b, d)
		upd(dict_c_d, c, d)

print(len(res_tests))
for test in res_tests:
	a, b, c, d = test[0], test[1], test[2], test[3]
	unique = 0
	unique += to_int(dict_a_b[hash(a, b)] == 1)
	unique += to_int(dict_a_c[hash(a, c)] == 1)
	unique += to_int(dict_a_d[hash(a, d)] == 1)
	unique += to_int(dict_b_c[hash(b, c)] == 1)
	unique += to_int(dict_b_d[hash(b, d)] == 1)
	unique += to_int(dict_c_d[hash(c, d)] == 1)
	print(chr(ord('A') + a), chr(ord('F') + b), chr(ord('K') + c), chr(ord('N') + d), end = '')
	print(' // %d unique pairs' % unique)
	
	