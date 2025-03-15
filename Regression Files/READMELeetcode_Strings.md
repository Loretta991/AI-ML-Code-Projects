
    <h1 style="color:#2E3A87; text-align:center;">Project Analysis</h1>
    <p style="color:#1F6B88; font-size:20px;">This project contains detailed analysis using Jupyter Notebooks. The following sections describe the steps, code implementations, and results.</p>
    <hr style="border: 2px solid #1F6B88;">
    
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            def containsDuplicate(nums):
    seen = set()
    for num in nums:
        if num in seen:
            return True
        seen.add(num)
    return False

# Example usage:
nums1 = [1, 2, 3, 1]
print(containsDuplicate(nums1))  # Output: True

nums2 = [1, 2, 3, 4]
print(containsDuplicate(nums2))  # Output: False

nums3 = [1, 1, 1, 3, 3, 4, 3, 2, 4, 2]
print(containsDuplicate(nums3))  # Output: True

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            class Solution:
    def containsDuplicate(self, nums):
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False

# Example usage:
solution = Solution()

nums1 = [1, 2, 3, 1]
result1 = solution.containsDuplicate(nums1)
print(result1)  # Output: True

nums2 = [1, 2, 3, 4]
result2 = solution.containsDuplicate(nums2)
print(result2)  # Output: False

nums3 = [1, 1, 1, 3, 3, 4, 3, 2, 4, 2]
result3 = solution.containsDuplicate(nums3)
print(result3)  # Output: True


            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            
#217
from typing import List

class Solution:
    def containsDuplicate(self, nums: List[int]) -> bool:
        """
        Check if any value appears at least twice in the array.

        :param nums: List of integers
        :return: True if any value is duplicated, False otherwise
        """
        seen = set()
        for num in nums:
            if num in seen:
                return True
            seen.add(num)
        return False

# Example usage:
solution = Solution()

nums1 = [1, 2, 3, 1]
result1 = solution.containsDuplicate(nums1)
print(result1)  # Output: True

nums2 = [1, 2, 3, 4]
result2 = solution.containsDuplicate(nums2)
print(result2)  # Output: False

nums3 = [1, 1, 1, 3, 3, 4, 3, 2, 4, 2]
result3 = solution.containsDuplicate(nums3)
print(result3)  # Output: True

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #228


"""

You are given a sorted unique integer array nums.

A range [a,b] is the set of all integers from a to b (inclusive).

Return the smallest sorted list of ranges that cover all the numbers in the array exactly.
That is, each element of nums is covered by exactly one of the ranges, and there is no integer x such
that x is in one of the ranges but not in nums.

Each range [a,b] in the list should be output as:

"a->b" if a != b
"a" if a == b


Example 1:

Input: nums = [0,1,2,4,5,7]
Output: ["0->2","4->5","7"]
Explanation: The ranges are:
[0,2] --> "0->2"
[4,5] --> "4->5"
[7,7] --> "7"
Example 2:

Input: nums = [0,2,3,4,6,8,9]
Output: ["0","2->4","6","8->9"]
Explanation: The ranges are:
[0,0] --> "0"
[2,4] --> "2->4"
[6,6] --> "6"
[8,9] --> "8->9"


Constraints:

0 <= nums.length <= 20
-231 <= nums[i] <= 231 - 1
All the values of nums are unique.
nums is sorted in ascending order.

"""
class Solution(object):
    def summaryRanges(self, nums):
        """
        :type nums: List[int]
        :rtype: List[str]
        """
        result = []
        if not nums:
            return result

        start, end = nums[0], nums[0]

        for num in nums[1:]:
            if num == end + 1:
                end = num
            else:
                result.append(self.format_range(start, end))
                start, end = num, num

        result.append(self.format_range(start, end))
        return result

    def format_range(self, start, end):
        if start == end:
            return str(start)
        else:
            return f"{start}->{end}"

# Example usage:
solution = Solution()

nums1 = [0, 1, 2, 4, 5, 7]
result1 = solution.summaryRanges(nums1)
print(result1)  # Output: ["0->2","4->5","7"]

nums2 = [0, 2, 3, 4, 6, 8, 9]
result2 = solution.summaryRanges(nums2)
print(result2)  # Output: ["0","2->4","6","8->9"]



            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #268
"""
Given an array nums containing n distinct numbers in the range [0, n], return the only number in the range that is missing from the array.



Example 1:

Input: nums = [3,0,1]
Output: 2
Explanation: n = 3 since there are 3 numbers, so all numbers are in the range [0,3]. 2 is the missing number in the range since it does not appear in nums.
Example 2:

Input: nums = [0,1]
Output: 2
Explanation: n = 2 since there are 2 numbers, so all numbers are in the range [0,2]. 2 is the missing number in the range since it does not appear in nums.
Example 3:

Input: nums = [9,6,4,2,3,5,7,0,1]
Output: 8
Explanation: n = 9 since there are 9 numbers, so all numbers are in the range [0,9]. 8 is the missing number in the range since it does not appear in nums.


Constraints:

n == nums.length
1 <= n <= 104
0 <= nums[i] <= n
All the numbers of nums are unique.


Follow up: Could you implement a solution using only O(1) extra space complexity and O(n) runtime complexity?


"""
#To solve this problem, you can calculate the sum of the first n natural numbers using the
#formula (n * (n + 1)) / 2. Then, subtract the sum of the given array from the expected sum. The result will be the missing number.
class Solution(object):
    def missingNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        expected_sum = (n * (n + 1)) // 2
        actual_sum = sum(nums)
        return expected_sum - actual_sum

# Example usage:
solution = Solution()

nums1 = [3, 0, 1]
result1 = solution.missingNumber(nums1)
print(result1)  # Output: 2

nums2 = [0, 1]
result2 = solution.missingNumber(nums2)
print(result2)  # Output: 2

nums3 = [9, 6, 4, 2, 3, 5, 7, 0, 1]
result3 = solution.missingNumber(nums3)
print(result3)  # Output: 8

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            
            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #283
#To solve this problem, you can iterate through the array and move all non-zero elements to the
#front of the array while maintaining
#their relative order. Then, fill the remaining positions with zero
class Solution(object):
    def moveZeroes(self, nums):
        """
        :type nums: List[int]
        :rtype: None Do not return anything, modify nums in-place instead.
        """
        non_zero_index = 0

        # Move non-zero elements to the front
        for num in nums:
            if num != 0:
                nums[non_zero_index] = num
                non_zero_index += 1

        # Fill the remaining positions with zeros
        for i in range(non_zero_index, len(nums)):
            nums[i] = 0

# Example usage:
solution = Solution()

nums1 = [0, 1, 0, 3, 12]
solution.moveZeroes(nums1)
print(nums1)  # Output: [1, 3, 12, 0, 0]

nums2 = [0]
solution.moveZeroes(nums2)
print(nums2)  # Output: [0]

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #303
"""
Given an integer array nums, handle multiple queries of the following type:

Calculate the sum of the elements of nums between indices left and right inclusive where left <= right.
Implement the NumArray class:

NumArray(int[] nums) Initializes the object with the integer array nums.
int sumRange(int left, int right) Returns the sum of the elements of nums between indices left and right inclusive (i.e. nums[left] + nums[left + 1] + ... + nums[right]).


Example 1:

Input
["NumArray", "sumRange", "sumRange", "sumRange"]
[[[-2, 0, 3, -5, 2, -1]], [0, 2], [2, 5], [0, 5]]
Output
[null, 1, -1, -3]

Explanation
NumArray numArray = new NumArray([-2, 0, 3, -5, 2, -1]);
numArray.sumRange(0, 2); // return (-2) + 0 + 3 = 1
numArray.sumRange(2, 5); // return 3 + (-5) + 2 + (-1) = -1
numArray.sumRange(0, 5); // return (-2) + 0 + 3 + (-5) + 2 + (-1) = -3


Constraints:

1 <= nums.length <= 104
-105 <= nums[i] <= 105
0 <= left <= right < nums.length
At most 104 calls will be made to sumRange.
"""
#This code initializes the NumArray object with the cumulative sum array in the __init__ method and calculates the sum of
#elements in the specified range using the sumRange method.

class NumArray(object):

    def __init__(self, nums):
        """
        Initialize the object with the integer array nums.
        :type nums: List[int]
        """
        self.cumulative_sum = [0] * (len(nums) + 1)
        for i in range(len(nums)):
            self.cumulative_sum[i + 1] = self.cumulative_sum[i] + nums[i]

    def sumRange(self, left, right):
        """
        Returns the sum of the elements of nums between indices left and right inclusive.
        :type left: int
        :type right: int
        :rtype: int
        """
        return self.cumulative_sum[right + 1] - self.cumulative_sum[left]

# Example usage:
numArray = NumArray([-2, 0, 3, -5, 2, -1])
print(numArray.sumRange(0, 2))  # Output: 1
print(numArray.sumRange(2, 5))  # Output: -1
print(numArray.sumRange(0, 5))  # Output: -3

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #350
"""
Given two integer arrays nums1 and nums2, return an array of their intersection. Each element in the result must appear as many times as it shows in both arrays and you may return the result in any order.



Example 1:

Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2,2]
Example 2:

Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
Output: [4,9]
Explanation: [9,4] is also accepted.


Constraints:

1 <= nums1.length, nums2.length <= 1000
0 <= nums1[i], nums2[i] <= 1000


Follow up:

What if the given array is already sorted? How would you optimize your algorithm?
What if nums1's size is small compared to nums2's size? Which algorithm is better?
What if elements of nums2 are stored on disk, and the memory is limited such that you cannot load all elements into the memory at once?

"""
#This code initializes a Solution class with the intersect method, which finds the
#intersection of two arrays with frequencies taken into account. The Counter class is
#used to count the occurrences of elements in nums1.

from collections import Counter

class Solution(object):
    def intersect(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: List[int]
        """
        result = []
        counter_nums1 = Counter(nums1)

        for num in nums2:
            if num in counter_nums1 and counter_nums1[num] > 0:
                result.append(num)
                counter_nums1[num] -= 1

        return result

# Example usage:
solution = Solution()

nums1 = [1, 2, 2, 1]
nums2 = [2, 2]
result1 = solution.intersect(nums1, nums2)
print(result1)  # Output: [2, 2]

nums3 = [4, 9, 5]
nums4 = [9, 4, 9, 8, 4]
result2 = solution.intersect(nums3, nums4)
print(result2)  # Output: [4, 9]

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #495

#To solve this problem, you can iterate through the timeSeries and calculate the total duration of poisoning.
#For each attack, check if it overlaps with the previous attack.
#If it does, only add the additional duration to the total.

#This code initializes a Solution class with the findPoisonedDuration method,
#which calculates the total duration of poisoning.
#It iterates through the timeSeries and considers the overlapping durations.

class Solution(object):
    def findPoisonedDuration(self, timeSeries, duration):
        """
        :type timeSeries: List[int]
        :type duration: int
        :rtype: int
        """
        if not timeSeries:
            return 0

        total_duration = duration
        for i in range(1, len(timeSeries)):
            gap = timeSeries[i] - timeSeries[i - 1]
            total_duration += min(gap, duration)

        return total_duration

# Example usage:
solution = Solution()

timeSeries1 = [1, 4]
duration1 = 2
result1 = solution.findPoisonedDuration(timeSeries1, duration1)
print(result1)  # Output: 4

timeSeries2 = [1, 2]
duration2 = 2
result2 = solution.findPoisonedDuration(timeSeries2, duration2)
print(result2)  # Output: 3

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #785
"""
There is an undirected graph with n nodes, where each node is numbered between 0 and n - 1. You are given a 2D array graph, where graph[u] is an array of nodes that node u is adjacent to. More formally, for each v in graph[u], there is an undirected edge between node u and node v. The graph has the following properties:

There are no self-edges (graph[u] does not contain u).
There are no parallel edges (graph[u] does not contain duplicate values).
If v is in graph[u], then u is in graph[v] (the graph is undirected).
The graph may not be connected, meaning there may be two nodes u and v such that there is no path between them.
A graph is bipartite if the nodes can be partitioned into two independent sets A and B such that every edge in the graph connects a node in set A and a node in set B.

Return true if and only if it is bipartite.



Example 1:


Input: graph = [[1,2,3],[0,2],[0,1,3],[0,2]]
Output: false
Explanation: There is no way to partition the nodes into two independent sets such that every edge connects a node in one and a node in the other.
Example 2:


Input: graph = [[1,3],[0,2],[1,3],[0,2]]
Output: true
Explanation: We can partition the nodes into two sets: {0, 2} and {1, 3}.


Constraints:

graph.length == n
1 <= n <= 100
0 <= graph[u].length < n
0 <= graph[u][i] <= n - 1
graph[u] does not contain u.
All the values of graph[u] are unique.
If graph[u] contains v, then graph[v] contains u.
"""
#To determine if a graph is bipartite, you can perform a depth-first search (DFS)
#or breadth-first search (BFS) and assign nodes to two sets (colors) such that adjacent
#nodes have different colors. If at any point, you encounter an edge between two nodes of
#the same color, the graph is not bipartite.

#This code initializes a Solution class with the isBipartite method, which uses DFS to check if the graph is bipartite.
#The colors array is used to assign colors to nodes during the traversal.

class Solution(object):
    def isBipartite(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: bool
        """
        n = len(graph)
        colors = [0] * n  # 0: not colored, 1: color A, -1: color B

        def dfs(node, color):
            if colors[node] != 0:
                return colors[node] == color

            colors[node] = color
            for neighbor in graph[node]:
                if not dfs(neighbor, -color):
                    return False
            return True

        for i in range(n):
            if colors[i] == 0 and not dfs(i, 1):
                return False

        return True

# Example usage:
solution = Solution()

graph1 = [[1, 2, 3], [0, 2], [0, 1, 3], [0, 2]]
result1 = solution.isBipartite(graph1)
print(result1)  # Output: False

graph2 = [[1, 3], [0, 2], [1, 3], [0, 2]]
result2 = solution.isBipartite(graph2)
print(result2)  # Output: True


            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #797

#in Python that aims for efficiency in both time and storage usage:
class Solution(object):
    def allPathsSourceTarget(self, graph):
        """
        :type graph: List[List[int]]
        :rtype: List[List[int]]
        """
        def dfs(node, path):
            if node == n - 1:
                result.append(path[:])
                return

            for neighbor in graph[node]:
                dfs(neighbor, path + [neighbor])

        result = []
        n = len(graph)
        dfs(0, [0])
        return result

# Example usage:
solution = Solution()

graph1 = [[1, 2], [3], [3], []]
result1 = solution.allPathsSourceTarget(graph1)
print(result1)  # Output: [[0, 1, 3], [0, 2, 3]]

graph2 = [[4, 3, 1], [3, 2, 4], [3], [4], []]
result2 = solution.allPathsSourceTarget(graph2)
print(result2)  # Output: [[0, 4], [0, 3, 4], [0, 1, 3, 4], [0, 1, 2, 3, 4], [0, 1, 4]]

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #841
#To achieve the fastest time and least storage usage, you can use an iterative approach with
#a stack for DFS and avoid using a set for visited rooms. Instead, you can mark visited rooms directly
#n the rooms array to save space. Here's an optimized implementation:

#This implementation uses an iterative DFS with a stack. The stack maintains the rooms to be visited,
#and the visited array directly marks whether a room has been visited or not. This avoids the overhead of using a set for visited rooms.

class Solution(object):
    def canVisitAllRooms(self, rooms):
        """
        :type rooms: List[List[int]]
        :rtype: bool
        """
        n = len(rooms)
        stack = [0]
        visited = [False] * n
        visited[0] = True

        while stack:
            current_room = stack.pop()
            for key in rooms[current_room]:
                if not visited[key]:
                    visited[key] = True
                    stack.append(key)

        return all(visited)

# Example usage:
solution = Solution()

rooms1 = [[1], [2], [3], []]
result1 = solution.canVisitAllRooms(rooms1)
print(result1)  # Output: True

rooms2 = [[1, 3], [3, 0, 1], [2], [0]]
result2 = solution.canVisitAllRooms(rooms2)
print(result2)  # Output: False

            </pre>


            
            <h3 style="color:#3C6A72;">Code Section:</h3>
            <pre style="background-color:#F4F6F9; padding:15px; border-radius:5px; border:1px solid #ddd; font-family:monospace; color:#3C6A72;">
            #997
"""
In a town, there are n people labeled from 1 to n. There is a rumor that one of these people is secretly the town judge.

If the town judge exists, then:

The town judge trusts nobody.
Everybody (except for the town judge) trusts the town judge.
There is exactly one person that satisfies properties 1 and 2.
You are given an array trust where trust[i] = [ai, bi] representing that the person labeled ai trusts the person labeled bi.
If a trust relationship does not exist in trust array, then such a trust relationship does not exist.

Return the label of the town judge if the town judge exists and can be identified, or return -1 otherwise.



Example 1:

Input: n = 2, trust = [[1,2]]
Output: 2
Example 2:

Input: n = 3, trust = [[1,3],[2,3]]
Output: 3
Example 3:

Input: n = 3, trust = [[1,3],[2,3],[3,1]]
Output: -1


Constraints:

1 <= n <= 1000
0 <= trust.length <= 104
trust[i].length == 2
All the pairs of trust are unique.
ai != bi
1 <= ai, bi <= n
"""
class Solution(object):
    def findJudge(self, n, trust):
        """
        :type n: int
        :type trust: List[List[int]]
        :rtype: int
        """
        if n == 1:
            return 1

        trust_counts = [0] * (n + 1)
        trusted_counts = [0] * (n + 1)

        for a, b in trust:
            trust_counts[a] += 1
            trusted_counts[b] += 1

        for i in range(1, n + 1):
            if trust_counts[i] == 0 and trusted_counts[i] == n - 1:
                return i

        return -1

# Example usage:
solution = Solution()

# Example 1
n1 = 2
trust1 = [[1, 2]]
result1 = solution.findJudge(n1, trust1)
print(result1)  # Output: 2

# Example 2
n2 = 3
trust2 = [[1, 3], [2, 3]]
result2 = solution.findJudge(n2, trust2)
print(result2)  # Output: 3

# Example 3
n3 = 3
trust3 = [[1, 3], [2, 3], [3, 1]]
result3 = solution.findJudge(n3, trust3)
print(result3)  # Output: -1

            </pre>


            
    <hr style="border: 2px solid #1F6B88;">
    <h3 style="color:#2E3A87;">Analysis and Results:</h3>
    <p style="color:#1F6B88; font-size:18px;">The notebook contains various steps for analyzing the dataset. Below you can see the results and analysis conducted during the notebook execution.</p>
    