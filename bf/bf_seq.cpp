// bf_seq.cpp
// g++ -O3 -std=c++17 bf_seq.cpp -o bf_seq
#include <bits/stdc++.h>
using namespace std;

static string read_all_trim(const string& path) {
    ifstream in(path, ios::binary);
    if (!in) { cerr << "Failed to open " << path << "\n"; exit(1); }
    string s((istreambuf_iterator<char>(in)), istreambuf_iterator<char>());
    while (!s.empty() && (s.back()=='\n' || s.back()=='\r')) s.pop_back();
    return s;
}

static long long bf_search_count(const string& text, const string& pat) {
    int n = (int)text.size();
    int m = (int)pat.size();
    if (m == 0 || n < m) return 0;

    long long count = 0;
    for (int i = 0; i <= n - m; i++) {
        int j = 0;
        while (j < m && text[i + j] == pat[j]) j++;
        if (j == m) count++;
    }
    return count;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cerr << "Usage: " << argv[0] << " <pattern_file> <text_file>\n";
        return 1;
    }
    string pat = read_all_trim(argv[1]);
    string txt = read_all_trim(argv[2]);

    auto t0 = chrono::high_resolution_clock::now();
    long long matches = bf_search_count(txt, pat);
    auto t1 = chrono::high_resolution_clock::now();

    double sec = chrono::duration<double>(t1 - t0).count();
    cout << "Matches: " << matches << "\n";
    cout << "Time(s): " << fixed << setprecision(6) << sec << "\n";
    return 0;
}
