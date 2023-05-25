#include <fstream>
#include <iostream>
#include <map>
using namespace std;

int main(int argc, char **argv)
{
	if (argc < 2) {
		cout << "Please pass the test name as first argument" << endl;
		exit(1);
	}
	// Open the graph file and read from it
	ifstream graph_file(argv[1], ios_base::in);
	int graph_size;
	graph_file >> graph_size;
    int adj_count[graph_size] = {0};
    while (!graph_file.eof()) {
		string a;
		int start, end, w;
		graph_file >> a >> start >> end >> w;
        start--;
        end--;
		if (a == "a") {
			adj_count[start]++;
			adj_count[end]++;
		}
	}
    // Count them
    map<int, int> frequency_map;
    for (int count : adj_count) {
        if (frequency_map.find(count) != frequency_map.end()) {
            frequency_map[count]++;
        } else {
            frequency_map[count] = 1;
        }
    }
    // Print
    for (auto frequency : frequency_map) {
        cout << frequency.first << "," << frequency.second << endl;
    }
    return 0;
}