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
int pr[M];

vector<pair<int, pair<int, int> > > edges;
vector<int> g[M];


int get(int v) {
	if (v == pr[v]) return v;
	return get(pr[v]);
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
    freopen("input.txt", "r", stdin);
    //freopen("output.txt", "w", stdout);
#endif
	cin.tie(0);
	ios_base::sync_with_stdio(0);

	int n;
	cin >> n;
	for (int i = 0; i < n;i++) pr[i] = i;

	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cin >> a[i][j];
			if (i < j) {
				edges.pb({a[i][j], {i, j}});
			}
		}
	}

	sort(edges.begin(), edges.end());
	for (auto e : edges) {
		int s = e.second.first;
		int f = e.second.second;
		if ((int)g[s].size() <= 1 && (int)g[f].size() <= 1 && get(s) != get(f)) {
			g[s].pb(f);
			g[f].pb(s);
			
		//	cout << s + 1 << " " << f + 1 << endl;

			pr[get(s)] = get(f);
		}
	}

	int st = -1;
	for (int i = 0; i < n; i++) {
		if ((int) g[i].size() == 1) {
			st = i;
			break;
		}
	}
	assert(st != -1);

	vector<int> ans;
	int answ = 0;

	int pr = st;
	ans.pb(st);
	st = g[st][0];
	for (int i = 1; i < n; i++) {
		//cout << "pr=" << pr + 1 << ", st=" << st + 1<< endl;
		ans.pb(st);
		int u = g[st][0] == pr ? g[st][1] : g[st][0];
		pr = st;
		st = u;
	}

	



	for (int i =0; i < n;i++) {
		answ += a[ans[i]][ans[(i + 1) % n]];
	}

	cout << answ << endl;
	for (int i = 0; i < n + 1; i++) {
		cout << ans[i % n] + 1 << " ";
	}




	
    return 0;
}   