#include <iostream>
#include <cstdio>
#include <ctime>
#include <cassert>
#include <cmath>
#include <stack>
#include <set>
#include <map>
#include <vector>
#include <queue>
#include <algorithm>
#include <utility>
#include <cstdlib>
#include <cstring>
#include <string>
using namespace std;

#ifdef WIN32
    #define lld "%I64d"
#else
    #define lld "%lld"
#endif

#define mp make_pair
#define pb push_back
#define put(x) { cout << #x << " = "; cout << (x) << endl; }

typedef long long ll;
typedef long double ld;
typedef unsigned long long ull;
typedef double db;

const int M = 4e3 + 15;
const int Q = 1e9 + 7;

int a[M][M];


vector<pair<int, pair<int, int> > > edges;
vector<int> g[M];

int cnt_tl = 0; 
bool is_tl() {
	if (cnt_tl == 0) {
		return 1.0 * clock() / CLOCKS_PER_SEC > 0.8;
	}
	return false;
}

void dfs(int v, int p) {
	cout << v + 1 << " ";
	if (v == 0) return;
	for (auto u : g[v]) {
		if (u != p) {
			dfs(u, v);
		}
	}
}

int main() {
    srand(time(NULL));
#ifdef LOCAL

    freopen("input.txt", "w", stdout);
#endif
	cin.tie(0);
	ios_base::sync_with_stdio(0);

	int n = 20;
	for (int i = 0; i < n; i++) {
		for (int j = i + 1; j < n; j++) {
			a[i][j] = rand() % 100000;
			a[j][i] = a[i][j];
		}
	}
	cout << n << endl;
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cout << a[i][j] << " ";
		}
		cout << endl;
	}
	
    return 0;
}   