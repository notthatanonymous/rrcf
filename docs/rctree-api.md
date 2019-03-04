# RCTree API documentation

This section enumerates all the methods of the `RCTree` class

## Inserting and deleting points

<b>`insert_point`</b>`(point, index, tolerance=None)`
> Inserts a point into the tree, creating a new leaf with given index

*Parameters:*

| Argument | Type | Description |
-----------|------|--------------
| `point`     | numpy ndarray (1 x d) | Data point |
| `index`       | any hashable type | Identifier for new leaf in tree |
| `tolerance`    | float      | Tolerance for determining duplicate points |

*Returns:*

| Object | Type | Description |
-----------|------|--------------
| `leaf`     | `Leaf` instance | New leaf in tree |

```python
>>> tree = rrcf.RCTree()

# Insert a point
>>> x = np.random.randn(2)
>>> tree.insert_point(x, index=0)
```

## Leaf and Branch operations

<b>`map_leaves`</b>`(node, op=(lambda x: None), *args, **kwargs)`
> Traverse tree recursively, calling operation given by op on leaves

*Parameters:*

| Argument | Type | Description |
-----------|------|--------------
| `node`      | `Leaf` or `Branch` instance | Node in RCTree |
| `op`       | function | Function to call on each leaf (defaults to no-op) |
| `*args`    | any      | Positional arguments to `op` |
| `**kwargs` | any      | Keyword arguments to `op` |

*Returns:*

`None`

```python
# Use map_leaves to print leaves in postorder

>>> X = np.random.randn(10, 2)
>>> tree = RCTree(X)
>>> tree.map_leaves(tree.root, op=print)

Leaf(5)
Leaf(9)
Leaf(4)
Leaf(0)
Leaf(6)
Leaf(2)
Leaf(3)
Leaf(7)
Leaf(1)
Leaf(8)
```

<b>`map_branches`</b>`(node, op=(lambda x: None), *args, **kwargs)`
> Traverse tree recursively, calling operation given by op on branches

*Parameters:*

| Argument | Type | Description |
-----------|------|--------------
| `node`      | `Leaf` or `Branch` instance | Node in RCTree |
| `op`       | function | Function to call on each leaf (defaults to no-op) |
| `*args`    | any      | Positional arguments to `op` |
| `**kwargs` | any      | Keyword arguments to `op` |

*Returns:*

`None`

```python
# Use map_branches to collect all branches in a list

>>> X = np.random.randn(10, 2)
>>> tree = RCTree(X)
>>> branches = []
>>> tree.map_branches(tree.root, op=(lambda x, stack: stack.append(x)),
                      stack=branches)
>>> branches

[Branch(q=0, p=-0.53),
 Branch(q=0, p=-0.35),
 Branch(q=1, p=-0.67),
 Branch(q=0, p=-0.15),
 Branch(q=0, p=0.23),
 Branch(q=1, p=0.29),
 Branch(q=1, p=1.31),
 Branch(q=0, p=0.62),
 Branch(q=1, p=0.86)]
```