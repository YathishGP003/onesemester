import type { Metadata } from "next";
import Sidebar from "../components/sidebar";
import { CodeBlock } from "../components/code-block";

export const generateMetadata = (): Metadata => {
  return {
    title: "Leetcode Snippet",
    description: "Collection of miscellaneous resources.",
    twitter: {
      card: "summary_large_image",
      title: "Beyond One Semester",
      description: "Collection of miscellaneous resources.",
      images: ["https://i.imgur.com/wPZN4AU.png"],
    },
    openGraph: {
      siteName: "One Semester",
      title: "Leetcode Snippet",
      description: "Collection of miscellaneous resources.",
      images: [
        {
          url: "https://i.imgur.com/wPZN4AU.png",
          width: 800,
          height: 600,
        },
        {
          url: "https://i.imgur.com/wPZN4AU.png",
          width: 1800,
          height: 1600,
          alt: "One Semester Programming Roadmaps",
        },
      ],
    },
  };
};

export default function Leetcode() {
  return (
    <div className="flex flex-col min-h-screen items-center justify-center relative">
      <div className="flex w-full md:w-1/2 py-16">
        <Sidebar />
        <main className="flex-1 p-0 md:p-0 w-full md:pl-0 border-l border-neutral-700 ">
          <h1 className="text-3xl md:text-4xl font-bold mb-2 px-4 pt-2">
            Leetcode Patterns
          </h1>
          <p className="text-2xl px-4 text-emerald-400">
            1. Sliding Window Pattern
          </p>
          <p className="text-2xl px-4 py-2">
            Used when you need to inspect subarrays or substrings that involve
            consecutive elements.
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">Problem:</span> Find the maximum
            sum of a subarray of size{" "}
            <span className="text-emerald-400">'k'</span>
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`class Solution {
public:
    int maxSumSubarray(vector<int>& nums, int k) {
        int maxSum = 0, windowSum = 0;
        for (int i = 0; i < k; i++) {
            windowSum += nums[i];
        }
        maxSum = windowSum;

        for (int i = k; i < nums.size(); i++) {
            windowSum += nums[i] - nums[i - k];
            maxSum = max(maxSum, windowSum);
        }
        return maxSum;
    }
};`}
            />
          </div>
          <p className="text-2xl px-4 text-emerald-400">
            2. Two Pointers Pattern
          </p>
          <p className="text-2xl px-4 py-2">
            Efficient for problems involving pairs of elements or comparisons
            within an array.
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">Problem:</span> Find two numbers
            in a sorted array that add up to a target sum.
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`class Solution {
public:
    vector<int> twoSum(vector<int>& numbers, int target) {
        int left = 0, right = numbers.size() - 1;
        while (left < right) {
            int sum = numbers[left] + numbers[right];
            if (sum == target) {
                return {left + 1, right + 1};
            } else if (sum < target) {
                left++;  // move the left pointer to increase the sum
            } else {
                right--;  // move the right pointer to decrease the sum
            }
        }
        return {};
    }
};`}
            />
          </div>

          <p className="text-2xl px-4 text-emerald-400">
            3. Fast and Slow Pointers (Tortoise and Hare)
          </p>
          <p className="text-2xl px-4 py-2">
            Used to detect cycles in linked lists or arrays.
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">Problem:</span> Detect if a
            linked list has a cycle.
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`class Solution {
public:
    bool hasCycle(ListNode *head) {
        ListNode *slow = head, *fast = head;
        while (fast && fast->next) {
            slow = slow->next;
            fast = fast->next->next;
            if (slow == fast) {
                return true; // cycle detected
            }
        }
        return false;
    }
};`}
            />
          </div>

          <p className="text-2xl px-4 text-emerald-400">
            4. Merge Intervals Pattern
          </p>
          <p className="text-2xl px-4 py-2">
            Used when working with overlapping intervals, such as scheduling.
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">Problem:</span> Merge overlapping
            intervals.
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end());  // sort by start time
        vector<vector<int>> merged;

        for (const auto& interval : intervals) {
            if (merged.empty() || merged.back()[1] < interval[0]) {
                merged.push_back(interval);  // no overlap, add the interval
            } else {
                merged.back()[1] = max(merged.back()[1], interval[1]);  // merge intervals
            }
        }
        return merged;
    }
};`}
            />
          </div>

          <p className="text-2xl px-4 text-emerald-400">
            5. Binary Search Pattern
          </p>
          <p className="text-2xl px-4 py-2">
            Ideal when searching for a target in a sorted array or a monotonic
            function.
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">Problem:</span> Find the first
            position of a target in a sorted array.
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`class Solution {
public:
    int search(vector<int>& nums, int target) {
        int left = 0, right = nums.size() - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                return mid;
            } else if (nums[mid] < target) {
                left = mid + 1;
            } else {
                right = mid - 1;
            }
        }
        return -1;
    }
};`}
            />
          </div>

          <p className="text-2xl px-4 text-emerald-400">
            6. Backtracking Pattern
          </p>
          <p className="text-2xl px-4 py-2">
            Used for problems that involve exploring all possible solutions,
            such as permutations, combinations, or subsets.
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">Problem:</span> Generate all
            subsets of a set.
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`class Solution {
public:
    void backtrack(vector<int>& nums, vector<vector<int>>& result, vector<int>& subset, int index) {
        result.push_back(subset);  // add current subset to the result
        for (int i = index; i < nums.size(); i++) {
            subset.push_back(nums[i]);  // include nums[i]
            backtrack(nums, result, subset, i + 1);  // recursively explore further
            subset.pop_back();  // exclude nums[i], backtrack to previous state
        }
    }

    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> result;
        vector<int> subset;
        backtrack(nums, result, subset, 0);
        return result;
    }
};`}
            />
          </div>

          <p className="text-2xl px-4 text-emerald-400">
            7. Dynamic Programming (DP) Pattern
          </p>
          <p className="text-2xl px-4 py-2">
            Used when a problem can be broken down into overlapping subproblems,
            with optimal substructure.
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">Problem:</span> Solve the 0/1
            Knapsack problem.
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`class Solution {
public:
    int knapsack(vector<int>& weights, vector<int>& values, int W) {
        int n = weights.size();
        vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));

        for (int i = 1; i <= n; i++) {
            for (int w = 1; w <= W; w++) {
                if (weights[i - 1] <= w) {
                    dp[i][w] = max(dp[i - 1][w], values[i - 1] + dp[i - 1][w - weights[i - 1]]);
                } else {
                    dp[i][w] = dp[i - 1][w];
                }
            }
        }
        return dp[n][W];
    }
};`}
            />
          </div>

          <p className="text-2xl px-4 text-emerald-400">8. Greedy Pattern</p>
          <p className="text-2xl px-4 py-2">
            Used when making local optimal choices at each step leads to a
            global optimal solution.
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">Problem:</span> Find the minimum
            number of coins to make up a target amount.
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        sort(coins.rbegin(), coins.rend());  // sort coins in descending order
        int count = 0;
        for (int coin : coins) {
            if (amount >= coin) {
                count += amount / coin;  // use as many large coins as possible
                amount %= coin;
            }
        }
        return amount == 0 ? count : -1;  // return -1 if the amount cannot be made
    }
};`}
            />
          </div>

          <p className="text-2xl px-4 text-emerald-400">
            9. Topological Sort (Graph Traversal)
          </p>
          <p className="text-2xl px-4 py-2">
            Used for problems involving dependency resolution, such as
            scheduling tasks.
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">Problem:</span> Find the order of
            courses to take given prerequisites.
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`class Solution {
public:
    vector<int> findOrder(int numCourses, vector<vector<int>>& prerequisites) {
        vector<int> indegree(numCourses, 0);
        vector<vector<int>> graph(numCourses);

        for (const auto& pre : prerequisites) {
            graph[pre[1]].push_back(pre[0]);
            indegree[pre[0]]++;
        }

        queue<int> q;
        for (int i = 0; i < numCourses; i++) {
            if (indegree[i] == 0) q.push(i);  // start with courses having no prerequisites
        }

        vector<int> order;
        while (!q.empty()) {
            int course = q.front();
            q.pop();
            order.push_back(course);
            for (int next : graph[course]) {
                if (--indegree[next] == 0) {
                    q.push(next);
                }
            }
        }

        return order.size() == numCourses ? order : vector<int>();  // return an empty array if a cycle exists
    }
};`}
            />
          </div>

          <p className="text-2xl px-4 text-emerald-400">
            10. Union-Find (Disjoint Set Union) Pattern
          </p>
          <p className="text-2xl px-4 py-2">
            Used for problems involving connected components in graphs.
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">Problem:</span> Detect if a graph
            contains a cycle.
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`class Solution {
public:
    int findParent(vector<int>& parent, int node) {
        if (parent[node] != node) {
            parent[node] = findParent(parent, parent[node]);  // path compression
        }
        return parent[node];
    }

    bool unionNodes(vector<int>& parent, vector<int>& rank, int u, int v) {
        int rootU = findParent(parent, u);
        int rootV = findParent(parent, v);
        if (rootU == rootV) return false;  // cycle detected

        if (rank[rootU] > rank[rootV]) {
            parent[rootV] = rootU;
        } else if (rank[rootU] < rank[rootV]) {
            parent[rootU] = rootV;
        } else {
            parent[rootV] = rootU;
            rank[rootU]++;
        }
        return true;
    }

    bool hasCycle(int n, vector<vector<int>>& edges) {
        vector<int> parent(n), rank(n, 0);
        iota(parent.begin(), parent.end(), 0);  // initialize each node as its own parent

        for (const auto& edge : edges) {
            if (!unionNodes(parent, rank, edge[0], edge[1])) {
                return true;  // cycle found
            }
        }
        return false;  // no cycle
    }
};`}
            />
          </div>

          <p className="text-2xl px-4 text-emerald-400">
            11. Cyclic Sort Pattern
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">When to Use:</span> When you're
            dealing with a range of numbers from 1 to n or 0 to n, and you're
            asked to <span className="text-emerald-400">rearrange</span> or{" "}
            <span className="text-emerald-400">
              find missing/duplicate numbers.
            </span>
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">Code Idea:</span>
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`vector<int> cyclicSort(vector<int>& nums) {
    int i = 0;
    while (i < nums.size()) {
        int correctIndex = nums[i] - 1;
        if (nums[i] != nums[correctIndex]) {
            swap(nums[i], nums[correctIndex]);
        } else {
            i++;
        }
    }
    return nums;
}`}
            />
            <p className="text-2xl px-4 text-emerald-400">Problems:</p>
            <ul className="list-disc text-2xl ml-4 px-4 py-2">
              <li>Find the Missing Number</li>
              <li>Find All Duplicates in an Array</li>
              <li>Set Mismatch</li>
            </ul>
          </div>

          <p className="text-2xl px-4 text-emerald-400">12. BFS/DFS Pattern</p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">When to Use:</span>
          </p>
          <p className="text-2xl px-4 py-2">
            When you're working with graphs or trees (or matrix as a grid).
          </p>
          <p className="text-2xl px-4 text-emerald-400">
            Data structures and Algorithms
          </p>
          <ul className="list-disc text-2xl ml-4 px-4 py-2">
            <li>Use DFS for deep traversal.</li>
            <li>Use BFS for shortest path in unweighted graphs.</li>
          </ul>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">DFS Code (Recursive):</span>
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`void dfs(int node, vector<vector<int>>& graph, vector<bool>& visited) {
    visited[node] = true;
    for (int neighbor : graph[node]) {
        if (!visited[neighbor])
            dfs(neighbor, graph, visited);
    }
}`}
            />
          </div>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">BFS Code:</span>
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`void bfs(int start, vector<vector<int>>& graph) {
    queue<int> q;
    vector<bool> visited(graph.size(), false);
    q.push(start);
    visited[start] = true;

    while (!q.empty()) {
        int node = q.front(); q.pop();
        for (int neighbor : graph[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}`}
            />
            <p className="text-2xl px-4 text-emerald-400">Problems:</p>
            <ul className="list-disc text-2xl ml-4 px-4 py-2">
              <li>Clone Graph</li>
              <li>Number of Provinces</li>
              <li>Word Ladder</li>
            </ul>
          </div>

          <p className="text-2xl px-4 text-emerald-400">
            13. Island (Matrix traversal) Pattern
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">When to Use:</span> When you are
            given a 2D grid and need to count or explore connected components.
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">DFS in Matrix:</span>
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`void dfs(vector<vector<char>>& grid, int r, int c) {
    if (r < 0 || c < 0 || r >= grid.size() || c >= grid[0].size() || grid[r][c] == '0')
        return;

    grid[r][c] = '0'; // mark visited

    dfs(grid, r+1, c);
    dfs(grid, r-1, c);
    dfs(grid, r, c+1);
    dfs(grid, r, c-1);
}`}
            />
            <p className="text-2xl px-4 text-emerald-400">Problems:</p>
            <ul className="list-disc text-2xl ml-4 px-4 py-2">
              <li>Number of Islands</li>
              <li>Surrounded Regions</li>
              <li>Flood Fill</li>
            </ul>
          </div>

          <p className="text-2xl px-4 text-emerald-400">
            14. In-place Reversal of Linked List
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">When to Use:</span> To reverse a
            linked list or part of it, especially in O(1) space.
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">Code:</span>
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`ListNode* reverseList(ListNode* head) {
    ListNode* prev = NULL;
    while (head) {
        ListNode* nextNode = head->next;
        head->next = prev;
        prev = head;
        head = nextNode;
    }
    return prev;
}`}
            />
            <p className="text-2xl px-4 text-emerald-400">Problems:</p>
            <ul className="list-disc text-2xl ml-4 px-4 py-2">
              <li>Reverse Linked List</li>
              <li>Reverse Nodes in k-Group</li>
              <li>Palindrome Linked List</li>
            </ul>
          </div>

          <p className="text-2xl px-4 text-emerald-400">
            15. Two Heaps Pattern
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">When to Use:</span> Used in
            problems where you need to find median, or maintain order statistics
            dynamically.
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">DFS in Matrix:</span>
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`priority_queue<int> maxHeap; // for left half
priority_queue<int, vector<int>, greater<int>> minHeap; // for right half

void addNum(int num) {
    if (maxHeap.empty() || num <= maxHeap.top()) maxHeap.push(num);
    else minHeap.push(num);

    // balance
    if (maxHeap.size() > minHeap.size() + 1) {
        minHeap.push(maxHeap.top()); maxHeap.pop();
    } else if (minHeap.size() > maxHeap.size()) {
        maxHeap.push(minHeap.top()); minHeap.pop();
    }
}`}
            />
            <p className="text-2xl px-4 text-emerald-400">Problems:</p>
            <ul className="list-disc text-2xl ml-4 px-4 py-2">
              <li>Median of Data Stream</li>
              <li>Sliding Window Median</li>
            </ul>
          </div>

          <p className="text-2xl px-4 text-emerald-400">
            16. Modified Binary Search Pattern
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">When to Use:</span> When you do
            binary search not on the values, but on a condition, index, or to
            solve min/max problems.
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">Code:</span>
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`int binarySearch(vector<int>& nums, int target) {
    int left = 0, right = nums.size() - 1;
    while (left <= right) {
        int mid = left + (right-left)/2;
        if (nums[mid] == target) return mid;
        else if (nums[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}`}
            />
            <p className="text-2xl px-4 py-2">
              <span className="text-emerald-400">Modified use case:</span> Find
              smallest number ≥ target.
            </p>
            <p className="text-2xl px-4 text-emerald-400">Problems:</p>
            <ul className="list-disc text-2xl ml-4 px-4 py-2">
              <li>Search in Rotated Array</li>
              <li>Peak Element</li>
              <li>Koko Eating Bananas</li>
            </ul>
          </div>
          <p className="text-2xl px-4 text-emerald-400">17. Top K Elements</p>
          <p className="text-2xl px-4 py-2">
            When you need to find k largest/smallest elements or frequent
            elements.
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">Code:</span>
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`vector<int> topKFrequent(vector<int>& nums, int k) {
    unordered_map<int, int> freq;
    for (int num : nums) freq[num]++;

    priority_queue<pair<int, int>> maxHeap;
    for (auto& [num, count] : freq)
        maxHeap.push({count, num});

    vector<int> res;
    while (k--) {
        res.push_back(maxHeap.top().second);
        maxHeap.pop();
    }
    return res;
}`}
            />
            <p className="text-2xl px-4 text-emerald-400">Problems:</p>
            <ul className="list-disc text-2xl ml-4 px-4 py-2">
              <li>Top K Frequent Elements</li>
              <li>K Closest Numbers</li>
              <li>Top K Frequent Words</li>
            </ul>
          </div>
          <p className="text-2xl px-4 text-emerald-400">
            18. 0/1 Knapsack Pattern
          </p>
          <p className="text-2xl px-4 py-2">
            Used in DP when choosing to either include or exclude an item.
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">Code:</span>
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`int knapsack(vector<int>& weight, vector<int>& value, int W) {
    int n = weight.size();
    vector<vector<int>> dp(n+1, vector<int>(W+1, 0));

    for (int i=1; i<=n; i++) {
        for (int w=0; w<=W; w++) {
            if (weight[i-1] <= w)
                dp[i][w] = max(dp[i-1][w], dp[i-1][w - weight[i-1]] + value[i-1]);
            else
                dp[i][w] = dp[i-1][w];
        }
    }
    return dp[n][W];
}`}
            />
            <p className="text-2xl px-4 text-emerald-400">Problems:</p>
            <ul className="list-disc text-2xl ml-4 px-4 py-2">
              <li>Subset Sum</li>
              <li>Partition Equal Subset Sum</li>
              <li>Target Sum</li>
            </ul>
          </div>
          <p className="text-2xl px-4 text-emerald-400">19. Bitwise XOR</p>
          <p className="text-2xl px-4 py-2">
            When you need to cancel out duplicate values or perform toggling.
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">Code:</span>
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`int findSingleNumber(vector<int>& nums) {
    int res = 0;
    for (int num : nums) res ^= num;
    return res;
}`}
            />
            <p className="text-2xl px-4 text-emerald-400">Problems:</p>
            <ul className="list-disc text-2xl ml-4 px-4 py-2">
              <li>Single Number</li>
              <li>Find Missing Number</li>
              <li>XOR of All Subsets</li>
            </ul>
          </div>
          <p className="text-2xl px-4 text-emerald-400">20. K-way Merge</p>
          <p className="text-2xl px-4 py-2">
            When merging k sorted arrays or lists.
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">Code:</span>
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`struct Compare {
    bool operator()(pair<int, pair<int, int>>& a, pair<int, pair<int, int>>& b) {
        return a.first > b.first;
    }
};

vector<int> mergeKSortedLists(vector<vector<int>>& lists) {
    priority_queue<pair<int, pair<int, int>>, vector<pair<int, pair<int, int>>>, Compare> pq;
    vector<int> indices(lists.size(), 0), result;

    for (int i = 0; i < lists.size(); i++) {
        if (!lists[i].empty())
            pq.push({lists[i][0], {i, 0}});
    }

    while (!pq.empty()) {
        auto [val, pos] = pq.top(); pq.pop();
        result.push_back(val);
        int i = pos.first, j = pos.second + 1;
        if (j < lists[i].size()) pq.push({lists[i][j], {i, j}});
    }

    return result;
}`}
            />
            <p className="text-2xl px-4 text-emerald-400">Problems:</p>
            <ul className="list-disc text-2xl ml-4 px-4 py-2">
              <li>Merge K Sorted Lists</li>
              <li>Smallest Range Covering Elements from K Lists</li>
            </ul>
          </div>
          <p className="text-2xl px-4 text-emerald-400">21. Monotonic Stack</p>
          <p className="text-2xl px-4 py-2">
            Used to find next/previous greater/smaller elements
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">Code:</span>
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`vector<int> nextGreater(vector<int>& nums) {
    vector<int> res(nums.size(), -1);
    stack<int> st;

    for (int i = 0; i < nums.size(); i++) {
        while (!st.empty() && nums[i] > nums[st.top()]) {
            res[st.top()] = nums[i];
            st.pop();
        }
        st.push(i);
    }

    return res;
}`}
            />
            <p className="text-2xl px-4 text-emerald-400">Problems:</p>
            <ul className="list-disc text-2xl ml-4 px-4 py-2">
              <li>Daily Temperatures</li>
              <li>Largest Rectangle in Histogram</li>
              <li>Next Greater Element</li>
            </ul>
          </div>
          <p className="text-2xl px-4 text-emerald-400">22. Multi-threaded</p>
          <p className="text-2xl px-4 py-2">
            When optimizing tasks with parallelism or for handling concurrent
            operations (not core DSA but used in system design/interviews).
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">Example Idea (Java):</span>
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="java"
              code={`class MyRunnable implements Runnable {
    public void run() {
        System.out.println("Thread is running");
    }
}`}
            />
            <p className="text-2xl px-4 text-emerald-400">Problems:</p>
            <ul className="list-disc text-2xl ml-4 px-4 py-2">
              <li>Multi-threaded FizzBuzz</li>
              <li>Building H2O</li>
              <li>Print in Order</li>
            </ul>
          </div>
          <p className="text-2xl px-4 text-emerald-400">
            23. Kth Smallest/Largest Element
          </p>
          <p className="text-2xl px-4 py-2">
            Asked directly to find k-th smallest/largest, can use heaps or
            quickselect.
          </p>
          <p className="text-2xl px-4 py-2">
            <span className="text-emerald-400">Heap Code:</span>
          </p>
          <div className="text-2xl px-4 py-2">
            <CodeBlock
              language="cpp"
              code={`int findKthLargest(vector<int>& nums, int k) {
    priority_queue<int, vector<int>, greater<int>> minHeap;

    for (int num : nums) {
        minHeap.push(num);
        if (minHeap.size() > k) minHeap.pop();
    }
    return minHeap.top();
}`}
            />
            <p className="text-2xl px-4 text-emerald-400">Problems:</p>
            <ul className="list-disc text-2xl ml-4 px-4 py-2">
              <li>Kth Smallest Element in BST</li>
              <li>Kth Largest Element in Array</li>
              <li>Find K Closest Elements</li>
            </ul>
          </div>
        </main>
      </div>
    </div>
  );
}
