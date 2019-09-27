Clone
list
with random pointer
-----------------------------
"""
# Definition for a Node.
class Node(object):
    def __init__(self, val, next, random):
        self.val = val
        self.next = next
        self.random = random
"""


class Solution(object):
    def copyRandomList(self, head):
        """
        :type head: Node
        :rtype: Node
        """
        if head == None:
            return
        if head.next == None:
            new_node = Node(head.val, head.next, head.random)
            return new_node

        curr_node = head
        while (curr_node != None):
            # create new node
            clone_node = Node(0, None, None)

            # add the node in between
            clone_node.next = curr_node.next
            curr_node.next = clone_node
            clone_node.random = curr_node.random

            curr_node = curr_node.next.next

        # re arrange all nodes
        original = head
        clone = head.next

        curr_original = head
        curr_clone = head.next
        while (curr_clone.next != None):
            curr_original.next = curr_original.next.next
            curr_original = curr_original.next

            curr_clone.next = curr_clone.next.next
            curr_clone = curr_clone.next
        return clone

    def copyRandomList(self, head):
        if not head:
            return None
        p = head
        while p:
            node = Node(p.val, None, p.random)
            node.next = p.next
            p.next = node
            p = p.next.next
            # p = node.next
        p = head
        while p:
            if p.random:
                p.next.random = p.random.next
            p = p.next.next
        newhead = head.next
        pold = head
        pnew = newhead
        while pnew.next:
            pold.next = pnew.next
            pold = pold.next
            pnew.next = pold.next
            pnew = pnew.next
        pold.next = None
        pnew.next = None
        return newhead


hashmap
solution

-----------------------------------------------------------------------------------------
unique
pairs
count -
public
static
int
getUniquePairs(int[]
nums, int
target){
    Arrays.sort(nums);
int
i = 0;
int
j = nums.length - 1;
int
ans = 0;
while (i < j){
int sum = nums[i]+ nums[j];
if (sum < target){
i++;
} else if (sum > target){
j--;
} else {
ans++;
i++;
j--;
while (i < j & & nums[i] == nums[i - 1]){
i++;
}
while (i < j & & nums[j] == nums[j + 1]){
j--;
}
}
}
return ans;
}
// java
O(n)
public
static
int
getUniquePairsOpti(int[]
nums, int
target){
Set < Integer > seen = new
HashSet <> ();
Map < Integer, Integer > map = new
HashMap <> ();
int
ans = 0;
for (int num: nums){
if (map.containsKey(num)){
int key = map.get(num) * 10 + num;
if (! seen.contains(key)){
ans++;
seen.add(key);
}
} else {
map.put(target-num, num);
}
}
return ans;

}
-----------------------------------------------------------------------------------------
Merge
two
lists
- --------------

head = ListNode(0)
cur = head

while l1 and l2:

    if l1.val > l2.val:
        cur.next = l2
        l2 = l2.next

    else:
        cur.next = l1
        l1 = l1.next

    cur = cur.next

cur.next = l1 or l2

return head.next

--------------------------------------------

ListNode * mergeTwoLists(ListNode * l1, ListNode * l2)
{

if (NULL == l1)
return l2;
if (NULL == l2) return l1;

ListNode * head = NULL; // head
of
the
list
to
return

// find
first
element(can
use
dummy
node
to
put
this
part
inside
of
the
loop)
if (l1->val < l2->val)       {head = l1; l1 = l1->next;}
else {head = l2; l2 = l2->next;}

ListNode * p = head; // pointer
to
form
new
list

// I
use & & to
remove
extra
IF
from the loop

while (l1 & & l2){
if (l1->val < l2->val)   {p->next = l1; l1 = l1->next;}
else {p->next = l2; l2 = l2->next;}
p=p->next;
}

// add
the
rest
of
the
tail, done!
if (l1)  p->next=l1;
else p->next=l2;

return head;
}
-----------------------------------------------------------------------------------------

Subtree
of
a
tree
- ----------------

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None


def areIdentical(root1, root2):
    if root1 is None and root2 is None:
        return True
    if root1 is None or root2 is None:
        return False

    return ((root1.val == root2.val) and
            areIdentical(root1.left, root2.left) and
            areIdentical(root1.right, root2.right))


def isSubTree(t, s):
    if s is None:
        return True
    if t is None:
        return False
    if (areIdentical(t, s)):
        return True
    return (isSubTree(t.left, s) or isSubTree(t.right, s))


class Solution(object):
    def isSubtree(self, s, t):
        """
        :type s: TreeNode
        :type t: TreeNode
        :rtype: bool
        """
        return isSubTree(s, t)


-----------------------------------------------------------------------------------------

Search in a
2
D
sorted
matrix
------------------------------

if (matrix == None or len(matrix) < 1 or len(matrix[0]) < 1):
    return False

col = len(matrix[0]) - 1
row = 0
while (col >= 0 and row <= len(matrix) - 1):
    if (target == matrix[row][col]):
        return True
    elif (target < matrix[row][col]):
        col -= 1
    elif (target > matrix[row][col]):
        row += 1
return False

-----------------------------------------------------------------------------------------

Critical
connections
---------------------
public


class Bridges {
private int id;
private int[] ids;
private int[] low;
private List < Integer >[] graph;

public List < List < Integer >> findBridges(int n, int[][] edges) {
List < List < Integer >> bridges = new ArrayList <> ();
graph = buildGraph(n, edges);
ids = new int[n + 1];
low = new int[n + 1];
id = 1;
visit(1, -1, bridges);


return bridges;
}

private
void
visit(int
at, int
parent, List < List < Integer >> bridges) {
low[at] = ids[at] = id + +;
for (int to: graph[at]) {
if (to == parent)
    continue;
if (ids[to] == 0) {// not visited
visit(to, at, bridges);
low[at] = Math.min(low[at], low[to]);
if (ids[at] < low[to]) {
bridges.add(Arrays.asList(at, to));
}
} else {
low[at] = Math.min(low[at], ids[to]);
}
}
}

private
List < Integer > []
buildGraph(int
n, int[][]
edges) {
    List < Integer > []
graph = new
List[n + 1];
Arrays.setAll(graph, (i) -> new
ArrayList <> ());
for (int[] edge: edges) {
    int
    u = edge[0];
int
v = edge[1];
graph[u].add(v);
graph[v].add(u);
}
return graph;
}
}

----------------------------------------------------------------------------------------- --
2
sum
Unique
pairs
- -------------
// Java
O(nlogn)
public
static
int
getUniquePairs(int[]
nums, int
target){
    Arrays.sort(nums);
int
i = 0;
int
j = nums.length - 1;
int
ans = 0;
while (i < j){
int sum = nums[i]+ nums[j];
if (sum < target){
i++;
} else if (sum > target){
j--;
} else {
ans++;
i++;
j--;
while (i < j & & nums[i] == nums[i - 1]){
i++;
}
while (i < j & & nums[j] == nums[j + 1]){
j--;
}
}
}
return ans;
}
// java
O(n)
public
static
int
getUniquePairsOpti(int[]
nums, int
target){
Set < Integer > seen = new
HashSet <> ();
Map < Integer, Integer > map = new
HashMap <> ();
int
ans = 0;
for (int num: nums){
if (map.containsKey(num)){
int key = map.get(num) * 10 + num;
if (! seen.contains(key)){
ans++;
seen.add(key);
}
} else {
map.put(target-num, num);
}
}
return ans;

}

---------------------------------------------------------------------------------------
min
max
value \
- ------------

public


class Solution {
public int uniquePathsWithObstacles(int[][] obstacleGrid) {

// Empty case
if (obstacleGrid.length == 0)


return 0;

int
rows = obstacleGrid.length;
int
cols = obstacleGrid[0].length;

for (int i = 0; i < rows; i++){
for (int j = 0; j < cols; j++){
if (obstacleGrid[i][j] == 1)
obstacleGrid[i][j] = 0;
else if (i == 0 & & j == 0)
obstacleGrid[i][j] = 1;
else if (i == 0)
obstacleGrid[i][j] = obstacleGrid[i][j - 1] * 1; // For row 0, if there are no paths to left cell, then its 0, else 1
else if (j == 0)
obstacleGrid[i][j] = obstacleGrid[i - 1][j] * 1; // For col 0, if there are no paths to upper cell, then its 0, else 1
else
obstacleGrid[i][j] = obstacleGrid[i - 1][j] + obstacleGrid[i][j - 1];
}
}

return obstacleGrid[rows - 1][cols - 1];

}
}

// replace
addition
with min()
     // find the overall max

at Most K different characters
-------------------------------


def subarraysWithKDistinct(self, A, K):
    return self.atMostK(A, K) - self.atMostK(A, K - 1)


def atMostK(self, A, K):
    count = collections.Counter()
    res = i = 0
    for j in range(len(A)):
        if count[A[j]] == 0: K -= 1
        count[A[j]] += 1
        while K < 0:
            count[A[i]] -= 1
            if count[A[i]] == 0: K += 1
            i += 1
        res += j - i + 1
    return res


----------------------------------------------------------------------------------------- -----------------------------------------------------------------------------------------

def countPairs(numCount, ratingValues, target):
    if numCount < 2:
        return 0

    complement = set()
    uniquePairs = set()

    for rating in ratingValues:
        val = target - rating
        if val in complement:
            pair = (rating, val) if rating > val else (val, rating)
            if pair not in uniquePairs:
                uniquePairs.add(pair)
        complement.add(rating)
    numDistinctPairs = len(uniquePairs)
    return numDistinctPairs
