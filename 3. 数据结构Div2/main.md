# STL

### 排序

```cpp
std::sort(a, a + n, &compare); //排序
```

```cpp
std::stable_sort(a, a + n, &compare); //稳定排序
```

```cpp
std::nth_element(a, a + pos, a + n, &compare) //使第pos个元素到排好序的位置上

template <class T>

struct greater {

bool operator()(const T& x, const T& y) const { return x > y; }

typedef T first_argument_type;

typedef T second_argument_type;

typedef bool result_type;

};
```

```
template <class T>

struct less {

bool operator()(const T& x, const T& y) const { return x < y; }

typedef T first_argument_type;

typedef T second_argument_type;

typedef bool result_type;

};
```

### 查找

```cpp
std::lower_bound(a, a + n, val, less<T>()); //返回第一个大于等于val的指针

std::upper_bound(a, a + n, val, less<T>()); //返回第一个大于val的指针

std::lower_bound(a, a + n, val, greater<T>()); //返回第一个小于等于val的指针

std::upper_bound(a, a + n, val, greater<T>()); //返回第一个小于val的指针
```

### 容器

### vector

```cpp
std::vector<T> vec; //动态数组，当预分配的空间大小不够时会重新分配空间
```

```cpp
vec.capacity(); //容器容量

vec.size(); //容器大小

vec.at(int idx); //用法和[]运算符相同

vec.push_back(); //尾部插入

vec.pop_back(); //尾部删除

vec.front(); //获取头部元素

vec.back(); //获取尾部元素

vec.begin(); //头元素的迭代器

vec.end(); //尾部元素的迭代器stack

vec.insert(pos, elem); // pos是vector的插入元素的位置

vec.insert(pos, n, elem); //在位置pos上插入n个元素elem

vec.insert(pos, begin, end);

vec.erase(pos); //移除pos位置上的元素，返回下一个数据的位置

vec.erase(begin, end); //移除[begin, end)区间的数据，返回下一个元素的位置

vec.reverse(pos1, pos2); //将vector中的pos1~pos2的元素逆序存储


Q.emplace(); //传参构造
```

### stack

```cpp
std::stack<T> sta; //栈，FILO

sta.empty(); //堆栈为空则返回真

sta.pop(); //移除栈顶元素

sta.push(); //在栈顶增加元素

sta.size(); //返回栈中元素数目

sta.top(); //返回栈顶元素，不删除

```

### queue



```cpp
std::queue<T> Q; //队列，FIFO，双端队列为 std::deque<T> Q;
```

```cpp
Q.front(); //返回 queue 中第一个元素的引用

Q.back(); //返回 queue 中最后一个元素的引用

Q.push(const T& obj); //在 queue 的尾部添加一个元素的副本

Q.push(T&& obj); //以移动的方式在 queue 的尾部添加元素

Q.pop(); //删除 queue 中的第一个元素

Q.size(); //返回 queue 中元素的个数

Q.empty(); //如果 queue 中没有元素的话，返回 true

Q.emplace(); //传参构造
```

### priority_queue

```cpp
std::priority_queue<T, vector<T>, less<T> > Q; //降序排列，大根堆

std::priority_queue<T, vector<T>, greater<T> > Q; //升序排列，小根堆
```

```cpp
Q.top(); //访问队头元素

Q.empty(); //队列是否为空

Q.size(); //返回队列内元素个数

Q.push(); //插入元素到队尾

Q.emplace(); //传参构造

Q.pop(); //弹出队头元素

Q.swap(); //交换内容
```

### set/multiset

```cpp
std::set<T, &compare> st; //不允许重复元素

std::multiset<T, &compare> st; //允许重复元素
```

```cpp
st.size(); //元素的数目

st.max_size(); //可容纳的最大元素的数量

st.empty(); //判断容器是否为空pair

st.find(elem); //返回值是迭代器类型

st.count(elem); // elem的个数，要么是1，要么是0，multiset可以大于一

st.begin(); //首元素的迭代器

st.end(); //尾后迭代器

st.rbegin(); //反向迭代器

st.rend(); //反向迭代器

st.insert(elem); //直接插入

st.insert(pos, elem); //指定位置插入

st.insert(begin, end); //插入一段迭代器指示位置

st.erase(pos); //删除迭代器指向的元素

st.erase(begin, end); //删除一段迭代器指示的元素

st.erase(elem); //删除一个元素

st.erase(st.find(elem)); // multiset删除一个元素

st.clear(); //清除所有元素
```

### pair

```cpp
std::pair<T1, T2> Pair; //二元组，支持列表初始化
```

```cpp
Pair.first;

Pair.second;
```

### map/multimap

```cpp
std::map<T1, T2> mp; //单重映射，允许下标访问

std::multimap<T1, T2> mp; //多重映射，不允许下标访问
```

```cpp
mp.at(key);

mp[key];

mp.count(key);

mp.max_size(); //求算容器最大存储量

mp.size(); //容器的大小

mp.begin();

mp.end();

mp.insert(elem);

mp.insert(pos, elem);

mp.insert(begin, end);

mp.erase(pos);

mp.erase(begin, end);

mp.erase(key);

mp.equal_range(key); // multimap使用，返回一对迭代器
```

### unordered_set/unordersd_map

```cpp
std::unordered_set<T> st; //无序set

std::unordersd_map<T1, T2> mp; //无序map

//操作有有序容器大致相同，但不支持迭代器遍历迭代器
```

### 迭代器

```cpp
std::CONTAINER<T>::iterator p; //迭代器 
std::CONTAINER<T>::reverse_iterator p; //反向迭代器
```

```cpp
for (std::CONTAINER<T>::iterator p = CONTAINER.begin(); 
p != CONTAINER.end(); p++) { 
} //正向遍历 
for (std::CONTAINER<T>::reverse_iterator p = CONTAINER.rbegin(); 
p != CONTAINER.rend(); p++) {
} //反向遍历
```

### 单调栈

顾名思义，单调栈即满足单调性的栈结构

例如，栈中自顶向下的元素为1,3,5,10,30,50, 插入元素20时为了保证单调性需要依次弹出元素1,3,5,10, 操作后栈变为20,30,50

[Problem - 1506](https://acm.hdu.edu.cn/showproblem.php?pid=1506)

```cpp
#include <bits/stdc++.h>
using namespace std;
const int maxn = 1e5 + 5;
#define ll long long int n;
int sta[maxn], top;
int a[maxn], l[maxn], r[maxn];
void Main()
{
    for (int i = 1; i <= n; i++)
        cin >> a[i];
    for (int i = 1; i <= n; i++)
        l[i] = r[i] = i;
    top = 0;
    for (int i = 1; i <= n; i++)
    {
        while (top && a[sta[top]] >= a[i])
            top--;
        if (!top)
            l[i] = 1;
        elsel[i] = sta[top] + 1;
        sta[++top] = i;
    }
    top = 0;
    for (int i = n; i >= 1; i--)
    {
        while (top && a[sta[top]] >= a[i])
            top--;
        if (!top)
            r[i] = n;
        elser[i] = sta[top] - 1;
        sta[++top] = i;
    }
    ll ans = 0;
    for (int i = 1; i <= n; i++)
        ans = max(ans, 1LL * a[i] * (r[i] - l[i] + 1));
    cout << ans << '\n';
}
int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    while (cin >> n && n)
        Main();
    return 0;
}
```

### 单调队列

[滑动窗口 /【模板】单调队列 - 洛谷](https://www.luogu.com.cn/problem/P1886)

题目：有一个长为n的序列 ，以及一个大小为k 的窗口；现在这个从左边开始向右滑动，每次滑动一个单位，求出每次滑动后窗口中的最大值和最小值?

对于最小值，我们维护一个单调上升的队列，每次插入值后判断队首元素是否在窗口内，若不在，弹出即可；最大值同理，维护一个单调下降的序列

```cpp
#include <bits/stdc++.h>
using namespace std;
const int maxn = 1e6 + 5;
int a[maxn];
deque<int> Q;
void Main()
{
    int n, k;
    cin >> n >> k;
    for (int i = 1; i <= n; i++)
        cin >> a[i];
    for (int i = 1; i <= n; i++)
    {
        while (!Q.empty() && a[Q.back()] >= a[i])
            Q.pop_back();
        Q.push_back(i);
        while (Q.front() < i - k + 1)
            Q.pop_front();
        if (i >= k)
            cout << a[Q.front()] << ' ';
    }
    cout << '\n';
    while (!Q.empty())
        Q.pop_back();
    for (int i = 1; i <= n; i++)
    {
        while (!Q.empty() && a[Q.back()] <= a[i])
            Q.pop_back();
        Q.push_back(i);
        while (Q.front() < i - k + 1)
            Q.pop_front();
        if (i >= k)
            cout << a[Q.front()] << ' ';
    }
}
int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    Main();
    return 0;
}
```

### ST表

[【模板】ST 表 - 洛谷](https://www.luogu.com.cn/problem/P3865)

给定个数，有m个询问，对于每个询问，你需要回答区间[l, r]中的最大值

```cpp
#include <bits/stdc++.h>
using namespace std;
const int maxn = 1e5 + 5;
const int maxd = 20;
int Log[maxn];
int f[maxn][maxd];
inline void init()
{
    for (int i = 2; i < maxn; i++)
        Log[i] = Log[i >> 1] + 1;
}
int calc(int l, int r)
{
    int s = Log[r - l + 1];
    return max(f[l][s], f[r - (1 << s) + 1][s]);
}
void Main()
{
    int n, m;
    cin >> n >> m;
    for (int i = 1; i <= n; i++)
        cin >> f[i][0];
    for (int j = 1; j < maxd; j++)
        for (int i = 1; i <= n; i++)
            if (i + (1 << (j - 1)) <= n)
                f[i][j] = max(f[i][j - 1], f[i + (1 << (j - 1))][j - 1]);
    for (int cas = 1; cas <= m; cas++)
    {
        int l, r;
        cin >> l >> r;
        cout << calc(l, r) << '\n';
    }
}
int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    init();
    Main();
    return 0;
}
```

### 并查集

[【模板】并查集 - 洛谷](https://www.luogu.com.cn/problem/P3367)

给出n个元素和m个操作

- 把两个元素所在的集合合并

- 查询两个元素是否在同一个集合内

- 查找
  
  对于同一个集合，他们的根节点一定相同，只需要一直向上找即可

- 合并
  
  合并两个集合的根节点即可

- 路径压缩
  
  我的祖先是谁与我父亲是谁没什么关系，这样一层一层找太浪费时间，不如我直接当祖先的儿子，问一次就可以出结果了，所以把路径上的每个节点都直接连接到根上
  
  
  
  ```cpp
  #include <bits/stdc++.h>
  using namespace std;
  const int maxn = 1e4 + 5;
  int fa[maxn];
  inline int find(int x)
  {
      if (x == fa[x])
          return x;
      return fa[x] = find(fa[x]);
  }
  inline void merge(int u, int v)
  {
      u = find(u), v = find(v);
      if (u == v)
          return;
      fa[u] = v;
  }
  void Main()
  {
      int n, m;
      cin >> n >> m;
      for (int i = 1; i <= n; i++)
          fa[i] = i;
      for (int i = 1; i <= m; i++)
      {
          int op, u, v;
          cin >> op >> u >> v;
          if (op == 1)
              merge(u, v);
          if (op == 2)
          {
              u = find(u), v = find(v);
              if (u == v)
                  cout << "Y\n";
              elsecout << "N\n";
          }
      }
  }
  int main()
  {
      ios::sync_with_stdio(0);
      cin.tie(0);
      cout.tie(0);
      Main();
      return 0;
  }
  ```

### 二叉堆

[【模板】堆 - 洛谷](https://www.luogu.com.cn/problem/P3378)

二叉堆是基于完全二叉树的基础上，加以一定的条件约束的一种特殊的二叉树

- 大根堆：任何一个父节点的值，都大于等于它左右孩子节点的值

- 小根堆：任何一个父节点的值，都小于等于它左右孩子节点的值

**二叉堆的相关操作**

- 添加节点
  
  最下一层最右边的叶子之后添加，如果这个结点的权值大于它父亲的权值，就交换，重复此过程直到不满足或者到根（向上调整）

- 删除节点
  
  根结点和最后一个结点直接交换，直接删掉根结点；在该结点的儿子中，找一个最大的，与该结点交换，重复此过程直到底层（向下调整）

- 实现
  
  使用一个序列 来表示堆， 的两个儿子分别是h2i和h2i+1， 是根结点

- 建堆
  
  - 向上调整
    
    $$
    log1+log2+···+log n = O(nlogn)
    $$
  
  - 向下调整
    
    $$
    nlogn-log1-log2-···-logn<=nlogn-0*2^0-1*2^1-···-(logn-1)*n/2
    =nlogn-(n-1)-(n-2)-(n-4)-···-(n-n/2)
    =nlogn-nlogn+1+2+4+···+n/2
    =n-1
    =O(n)
    $$
  
  ```cpp
  #include <bits/stdc++.h>
  using namespace std;
  const int maxn = 1e6 + 5;
  struct Heap
  {
      int siz, h[maxn];
      void insert(int x)
      {
          h[++siz] = x;
          int now = siz;
          while (now > 1 && h[now] < h[now >> 1])
          {
              swap(h[now], h[now >> 1]);
              now >>= 1;
          }
      }
      void pop()
      {
          h[1] = h[siz--];
          int now = 1;
          while (now << 1 <= siz)
          {
              int son = now << 1;
              if (son < siz && h[son | 1] < h[son])
                  son |= 1;
              if (h[now] <= h[son])
                  return;
              swap(h[now], h[son]);
              now = son;
          }
      }
      int top() { return h[1]; }
  } heap;
  void Main()
  {
      int n;
      cin >> n;
      for (int i = 1; i <= n; i++)
      {
          int op;
          cin >> op;
          if (op == 1)
          {
              int x;
              cin >> x;
              heap.insert(x);
          }
          if (op == 2)
              cout << heap.top() << '\n';
          if (op == 3)
              heap.pop();
      }
  }
  int main()
  {
      ios::sync_with_stdio(0);
      cin.tie(0);
      cout.tie(0);
      Main();
      return 0;
  }
  ```

### 树状数组

```cpp
inline int lowbit(int x) { 
    return x & (-x); // lowbit(0b10100)=0b100 
}
```

```cpp
inline void add(int x, int val) { 
    while (x <= n) 
        c[x] += val, x += lowbit(x); 
}
```

```cpp
inline int query(int x) { 
    int res = 0; 
    while (x) 
        res += c[x], x -= lowbit(x); 
    return res; 
}
```



- 单调修改，区间查询

        [130. 树状数组 1 ：单点修改，区间查询](https://loj.ac/p/130)

```cpp
#include <bits/stdc++.h>
using namespace std;
const int maxn = 1e6 + 5;
#define ll long long 
struct BIT
{
    ll sum[maxn];
    inline int lowbit(int x) { return x & (-x); }
    void add(int x, int val)
    {
        while (x < maxn)
            sum[x] += val, x += lowbit(x);
    }
    inline ll query(int x)
    {
        ll res = 0;
        while (x)
            res += sum[x], x -= lowbit(x);
        return res;
    }
} T;
void Main()
{
    int n, m;
    cin >> n >> m;
    for (int i = 1; i <= n; i++)
    {
        int val;
        cin >> val;
        T.add(i, val);
    }
    for (int i = 1; i <= m; i++)
    {
        int op, a, b;
        cin >> op >> a >> b;
        if (op == 1)
            T.add(a, b);
        if (op == 2)
            cout << T.query(b) - T.query(a - 1) << '\n';
    }
}
int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    Main();
    return 0;
}
```

- 区间修改，单点查询

        [131. 树状数组 2 ： 区间修改，单点查询](https://loj.ac/p/131)

```cpp
#include <bits/stdc++.h>
using namespace std;
const int maxn = 1e6 + 5;
#define ll long long 
struct BIT
{
    ll sum[maxn];
    inline int lowbit(int x) { return x & (-x); }
    void add(int x, int val)
    {
        while (x < maxn)
            sum[x] += val, x += lowbit(x);
    }
    inline ll query(int x)
    {
        ll res = 0;
        while (x)
            res += sum[x], x -= lowbit(x);
        return res;
    }
} T;
void Main()
{
    int n, m;
    cin >> n >> m;
    for (int i = 1; i <= n; i++)
    {
        int val;
        cin >> val;
        T.add(i, val);
        T.add(i + 1, -val);
    }
    for (int i = 1; i <= m; i++)
    {
        int op;
        cin >> op;
        if (op == 1)
        {
            int l, r, val;
            cin >> l >> r >> val;
            T.add(l, val);
            T.add(r + 1, -val);
        }
        if (op == 2)
        {
            int x;
            cin >> x;
            cout << T.query(x) << '\n';
        }
    }
}
int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    Main();
    return 0;
}
```

- 区间修改，区间查询

        [132. 树状数组 3： 区间修改，区间查询](https://loj.ac/p/132)

```cpp
#include <bits/stdc++.h>
using namespace std;
const int maxn = 1e6 + 5;
#define ll long long 
struct BIT
{
    ll sum[maxn];
    inline int lowbit(int x) { return x & (-x); }
    void add(int x, ll val)
    {
        while (x < maxn)
            sum[x] += val, x += lowbit(x);
    }
    inline ll query(int x)
    {
        ll res = 0;
        while (x)
            res += sum[x], x -= lowbit(x);
        return res;
    }
} T1, T2;
inline void Add(int l, int r, int val)
{
    T1.add(l, val);
    T1.add(r + 1, -val);
    T2.add(l, 1LL * l * val);
    T2.add(r + 1, 1LL * (r + 1) * (-val));
}
inline ll Query(int x) { return 1LL * (x + 1) * T1.query(x) - T2.query(x); }
void Main()
{
    int n, m;
    cin >> n >> m;
    for (int i = 1; i <= n; i++)
    {
        int val;
        cin >> val;
        Add(i, i, val);
    }
    for (int i = 1; i <= m; i++)
    {
        int op;
        cin >> op;
        if (op == 1)
        {
            int l, r, val;
            cin >> l >> r >> val;
            Add(l, r, val);
        }
        if (op == 2)
        {
            int l, r;
            cin >> l >> r;
            cout << Query(r) - Query(l - 1) << '\n';
        }
    }
}
int main()
{
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);
    Main();
    return 0;
}
```

### 线段树

- **普通线段树**
  
  线段树将每个长度不为的区间划分成左右两个区间递归求解，把整个线段划分为一个树形结构，通过合并左右两区间信息来求得该区间的信息
  
  ```cpp
  #include <bits/stdc++.h>
  using namespace std;
  const int maxn = 1e6 + 5;
  #define ll long long 
  #define ls rt << 1 
  #define rs rt << 1 | 1 
  struct Tree
  {
      ll sum[maxn << 2];
      int L[maxn << 2], R[maxn << 2];
      void pushup(int rt) { sum[rt] = sum[ls] + sum[rs]; }
      void build(int rt, int l, int r)
      {
          L[rt] = l, R[rt] = r;
          if (l == r)
          {
              cin >> sum[rt];
              return;
          }
          int mid = (l + r) >> 1;
          build(ls, l, mid);
          build(rs, mid + 1, r);
          pushup(rt);
      }
      void add(int rt, int pos, int val)
      {
          if (L[rt] == R[rt])
          {
              sum[rt] += val;
              return;
          }
          int mid = (L[rt] + R[rt]) >> 1;
          if (pos <= mid)
              add(ls, pos, val);
          elseadd(rs, pos, val);
          pushup(rt);
      }
      ll query(int rt, int l, int r)
      {
          if (L[rt] == l && R[rt] == r)
              return sum[rt];
          int mid = (L[rt] + R[rt]) >> 1;
          if (r <= mid)
              return query(ls, l, r);
          if (l >= mid + 1)
              return query(rs, l, r);
          return query(ls, l, mid) + query(rs, mid + 1, r);
      }
  } T;
  void Main()
  {
      int n, m;
      cin >> n >> m;
      T.build(1, 1, n);
      for (int i = 1; i <= m; i++)
      {
          int op, a, b;
          cin >> op >> a >> b;
          if (op == 1)
              T.add(1, a, b);
          if (op == 2)
              cout << T.query(1, a, b) << '\n';
      }
  }
  int main()
  {
      ios::sync_with_stdio(0);
      cin.tie(0);
      cout.tie(0);
      Main();
      return 0;
  }
  ```

- **权值线段树**
  
  [逆序对 - 洛谷](https://www.luogu.com.cn/problem/P1908)
  
  ```cpp
  #include <bits/stdc++.h>
  using namespace std;
  const int maxn = 5e5 + 5;
  #define ll long long 
  #define ls rt << 1 
  #define rs rt << 1 | 1 
  struct Tree
  {
      int L[maxn << 2], R[maxn << 2], sum[maxn << 2];
      void pushup(int rt) { sum[rt] = sum[ls] + sum[rs]; }
      void build(int rt, int l, int r)
      {
          L[rt] = l, R[rt] = r;
          if (l == r)
              return;
          int mid = (l + r) >> 1;
          build(ls, l, mid);
          build(rs, mid + 1, r);
      }
      void add(int rt, int pos)
      {
          if (L[rt] == R[rt])
          {
              sum[rt]++;
              return;
          }
          int mid = (L[rt] + R[rt]) >> 1;
          if (pos <= mid)
              add(ls, pos);
          elseadd(rs, pos);
          pushup(rt);
      }
      int query(int rt, int l, int r)
      {
          if (l > r)
              return 0;
          if (L[rt] == l && R[rt] == r)
              return sum[rt];
          int mid = (L[rt] + R[rt]) >> 1;
          if (r <= mid)
              return query(ls, l, r);
          if (l >= mid + 1)
              return query(rs, l, r);
          return query(ls, l, mid) + query(rs, mid + 1, r);
      }
  } T;
  int a[maxn], b[maxn];
  void Main()
  {
      int n;
      cin >> n;
      for (int i = 1; i <= n; i++)
          cin >> a[i], b[i] = a[i];
      sort(b + 1, b + n + 1);
      int m = unique(b + 1, b + n + 1) - b - 1;
      T.build(1, 1, m);
      ll ans = 0;
      for (int i = 1; i <= n; i++)
      {
          int pos = lower_bound(b + 1, b + m + 1, a[i]) - b;
          ans += T.query(1, pos + 1, m);
          T.add(1, pos);
      }
      cout << ans;
  }
  int main()
  {
      ios::sync_with_stdio(0);
      cin.tie(0);
      cout.tie(0);
      Main();
      return 0;
  }
  ```

- **lazy标记**
  
  ```cpp
  #include <bits/stdc++.h>
  using namespace std;
  const int maxn = 1e6 + 5;
  #define ll long long 
  #define ls rt << 1 
  #define rs rt << 1 | 1 
  struct Tree
  {
      int L[maxn << 2], R[maxn << 2];
      ll sum[maxn << 2], lazy[maxn << 2];
      void pushup(int rt) { sum[rt] = sum[ls] + sum[rs]; }
      void pushdown(int rt)
      {
          if (!lazy[rt])
              return;
          lazy[ls] += lazy[rt], lazy[rs] += lazy[rt];
          sum[ls] += lazy[rt] * (R[ls] - L[ls] + 1);
          sum[rs] += lazy[rt] * (R[rs] - L[rs] + 1);
          lazy[rt] = 0;
      }
      void build(int rt, int l, int r)
      {
          L[rt] = l, R[rt] = r;
          if (l == r)
          {
              cin >> sum[rt];
              return;
          }
          int mid = (l + r) >> 1;
          build(ls, l, mid);
          build(rs, mid + 1, r);
          pushup(rt);
      }
      void add(int rt, int l, int r, int val)
      {
          if (L[rt] == l && R[rt] == r)
          {
              lazy[rt] += val;
              sum[rt] += 1LL * val * (r - l + 1);
              return;
          }
          pushdown(rt);
          int mid = (L[rt] + R[rt]) >> 1;
          if (r <= mid)
              add(ls, l, r, val);
          else if (l >= mid + 1)
              add(rs, l, r, val);
          elseadd(ls, l, mid, val), add(rs, mid + 1, r, val);
          pushup(rt);
      }
      ll query(int rt, int l, int r)
      {
          if (L[rt] == l && R[rt] == r)
              return sum[rt];
          pushdown(rt);
          int mid = (L[rt] + R[rt]) >> 1;
          if (r <= mid)
              return query(ls, l, r);
          if (l >= mid + 1)
              return query(rs, l, r);
          return query(ls, l, mid) + query(rs, mid + 1, r);
      }
  } T;
  void Main()
  {
      int n, m;
      cin >> n >> m;
      T.build(1, 1, n);
      for (int i = 1; i <= m; i++)
      {
          int op, l, r;
          cin >> op >> l >> r;
          if (op == 1)
          {
              int val;
              cin >> val;
              T.add(1, l, r, val);
          }
          if (op == 2)
              cout << T.query(1, l, r) << '\n';
      }
  }
  int main()
  {
      ios::sync_with_stdio(0);
      cin.tie(0);
      cout.tie(0);
      Main();
      return 0;
  }
  ```

- **lazy标记持久化**
  
  - 为什么需要标记永久化？
    
    当我们用可持久化的数据结构时lazy标记不能下传，或者用动态开点线段树时如果标记下传会造成无用空间
  
  ```cpp
  #include <bits/stdc++.h>
  using namespace std;
  const int maxn = 1e6 + 5;
  #define ll long long
  #define ls rt << 1
  #define rs rt << 1 | 1
  struct Tree
  {
      int L[maxn << 2], R[maxn << 2];
      ll sum[maxn << 2], tag[maxn << 2];
      void build(int rt, int l, int r)
      {
          L[rt] = l, R[rt] = r;
          if (l == r)
          {
              cin >> sum[rt];
              return;
          }
          int mid = (l + r) >> 1;
          build(ls, l, mid);
          build(rs, mid + 1, r);
          sum[rt] = sum[ls] + sum[rs];
      }
      void add(int rt, int l, int r, int val)
      {
          sum[rt] += 1LL * val * (r - l + 1);
          if (L[rt] == l && R[rt] == r)
          {
              tag[rt] += val;
              return;
          }
          int mid = (L[rt] + R[rt]) >> 1;
          if (r <= mid)
              add(ls, l, r, val);
          else if (l >= mid + 1)
      }
      ll query(int rt, int l, int r, ll res = 0)
      {
          if (L[rt] == l && R[rt] == r)
              return sum[rt] + res * (r - l + 1);
          res += tag[rt];
          int mid = (L[rt] + R[rt]) >> 1;
          if (r <= mid)
              return query(ls, l, r, res);
          if (l >= mid + 1)
              return query(rs, l, r, res);
          return query(ls, l, mid, res) + query(rs, mid + 1, r, res);
      }
  } T;
  void Main()
  {
      int n, m;
      cin >> n >> m;
      T.build(1, 1, n);
      for (int i = 1; i <= m; i++)
      {
          int op, l, r;
          cin >> op >> l >> r;
          if (op == 1)
          {
              int val;
              cin >> val;
              T.add(1, l, r, val);
          }
          if (op == 2)
              cout << T.query(1, l, r) << '\n';
      }
  }
  int main()
  {
      ios::sync_with_stdio(0);
      cin.tie(0);
      cout.tie(0);
      Main();
      return 0;
  }
  ```
  
  

- 动态开点线段树
  
  [逆序对 - 洛谷](https://www.luogu.com.cn/problem/P1908)
  
  在一些计数问题中，线段树用于维护值域（一段权值范围），这样的线段树也称为权值线段树，为了降低空间复杂度，我们可以不建出整棵线段树的结构，而是在最初只建立一个根节点，代表整个区间，当需要访问线段树的某棵子树（某个子区间）时，再建立代表这个子区间的节点，采用这种方法维护的线段树称为动态开点线段树
  
  ```cpp
  #include <bits/stdc++.h>
  using namespace std;
  const int maxn = 1e7 + 5;
  #define ll long long 
  struct Tree
  {
      int cnt, ls[maxn], rs[maxn], sum[maxn];
      void add(int &rt, int pos, int val, int L = 1, int R = 1e9)
      {
          if (!rt)
              rt = ++cnt;
          if (L == R)
          {
              sum[rt] += val;
              return;
          }
          int mid = (L + R) >> 1;
          if (pos <= mid)
              add(ls[rt], pos, val, L, mid);
          elseadd(rs[rt], pos, val, mid + 1, R);
          sum[rt] = sum[ls[rt]] + sum[rs[rt]];
      }
      int query(int rt, int l, int r, int L = 1, int R = 1e9)
      {
          if (!rt)
              return 0;
          if (l > r)
              return 0;
          if (l == L && r == R)
              return sum[rt];
          int mid = (L + R) >> 1;
          if (r <= mid)
              return query(ls[rt], l, r, L, mid);
          if (l >= mid + 1)
              return query(rs[rt], l, r, mid + 1, R);
          return query(ls[rt], l, mid, L, mid) + query(rs[rt], mid + 1, r, mid + 1, R);
      }
  }T;
  void Main()
  {
      int n;
      cin >> n;
      ll ans = 0;
      int rt = 0;
      for (int i = 1; i <= n; i++)
      {
          int x;
          cin >> x;
          ans += T.query(rt, x + 1, 1e9);
          T.add(rt, x, 1);
      }
      cout << ans;
  }
  int main()
  {
      ios::sync_with_stdio(0);
      cin.tie(0);
      cout.tie(0);
      Main();
      return 0;
  }
  ```

