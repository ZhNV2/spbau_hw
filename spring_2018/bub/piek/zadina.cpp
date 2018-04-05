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

int g[M][M];
int used[M];
int p[M];

int cnt_tl = 0;
bool is_tl() {
	cnt_tl = (cnt_tl + 1) % 1000;
	if (cnt_tl == 999) {
		return 1.0 * clock() / CLOCKS_PER_SEC > 0.3;
	}
	return false;
}

void try_improove(vector<int> &ans) {
	int n = (int) ans.size();
	int T = 6;
	int bst[10];

	while (!is_tl()) {
		int sh = rand() % n;
		rotate(ans.begin(), ans.begin() + sh, ans.end());
		int be = Q;
		
		sort(ans.begin() + 1, ans.begin() + T + 1);
		do {
			int cur = 0;
			for (int i = 0; i < T + 1; i++) {
				cur += g[ans[i]][ans[i + 1]];
			}
			if (cur < be) {
				be = cur;
				for (int i = 1; i < T + 1; i++) {
					bst[i] = ans[i];
				}
			}
		} while (next_permutation(ans.begin() + 1, ans.begin() + T + 1));
		for (int i= 1; i < T + 1; i++) {
			ans[i] = bst[i];
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
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			cin >> g[i][j];
		}
	}
	
	int answ = Q;
	vector<int> ans;

	if (n <= 10) {
		int p[10];
		for (int i = 0; i < n; i++) p[i] = i;
		do {
			int cur = 0;
			for (int i = 0; i < n; i++) {
				cur += g[p[i]][p[(i + 1) % n]];
			}
			if (cur < answ) {
				answ = cur;
				ans.resize(0);
				for (int i = 0; i < n; i++) {
					ans.pb(p[i]);
				}
			}
		} while (next_permutation(p, p + n));
	} else {
		for (int v = 0; v < n; v++) {
			int cur = 0;
			for (int i = 0;i < n;i++) used[i] = 0;
			used[v] = 1;
			p[0] = v;
			for (int i = 1; i < n; i++) {
				int pi = -1;
				for (int j = 0; j < n; j++) {
					if (!used[j] && (pi == -1 || g[p[i - 1]][j] < g[p[i - 1]][pi])) {
						pi = j;
					}
				}
				used[pi] = 1;
				p[i] = pi;
				cur += g[p[i - 1]][pi];
			}
			cur += g[p[n - 1]][v];
			if (cur < answ) {
				answ = cur;
				ans.resize(0);
				for (int i = 0; i < n; i++) {
					ans.pb(p[i]);
				}
			}
		}
		try_improove(ans);
	}


	
	

	cerr << answ << endl;

	

	answ = 0;
	for (int i =0; i < n;i++) {
		answ += g[ans[i]][ans[(i + 1) % n]];
	}

	cout << answ << endl;
	for (int i = 0; i < n + 1; i++) {
		cout << ans[i % n] + 1 << " ";
	}
	cout << endl;



	
    return 0;
}   