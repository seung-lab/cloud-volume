class DoublyLinkedListIterator(object):
  def __init__(self, node, reverse=False):
    self.node = ListNode(None, node, node)
    self.reverse = reverse
  def next(self):
    return self.__next__()
  def __next__(self):
    if self.reverse:
      if self.node.prev is not None:
        self.node = self.node.prev
        return self.node
    else:
      if self.node.next is not None:
        self.node = self.node.next
        return self.node
    raise StopIteration()
  def __reversed__(self):
    return DoublyLinkedListIterator(self.node, not self.reverse)

class ListNode(object):
  def __init__(self, val, next, prev):
    self.val = val
    self.next = next
    self.prev = prev 

  def __iter__(self):
    return DoublyLinkedListIterator(self)

  def __reversed__(self):
    return DoublyLinkedListIterator(self, reverse=True)

  def __str__(self):
    return "ListNode({},{},{})".format(
      self.val, 
      self.next.val if self.next is not None else None,
      self.prev.val if self.prev is not None else None
    )

  def clone(self):
    return ListNode(self.val, self.next, self.prev)

class DoublyLinkedList(object):
  def __init__(self):
    self.head = None
    self.tail = None
    self.size = 0

  @classmethod
  def create(cls, lst):
    lst = iter(lst)

    dll = DoublyLinkedList()
    for val in lst:
      dll.append(val)

    return dll

  def __len__(self):
    return self.size

  def __iter__(self):
    return DoublyLinkedListIterator(self.head)

  def __reversed__(self):
    return DoublyLinkedListIterator(self.tail, reverse=True)

  def tolist(self):
    return [ x.val for x in self ]

  def is_empty(self):
    return self.head is None and self.tail is None

  def peek_head(self):
    if self.head is None:
      return None
    return self.head.val

  def peek_tail(self):
    if self.tail is None:
      return None
    return self.tail.val

  def promote_to_head(self, node):
    self.delete(node)

    if self.head is None:
      self.head = node
      node.next = None 
      node.prev = None
      self.tail = node
    else:
      head = self.head
      head.prev = node
      self.head = node
      node.prev = None 
      node.next = head

    self.size += 1

  def delete(self, node):
    nxt, prev = node.next, node.prev

    if prev is not None:
      prev.next = nxt

    if nxt is not None:
      nxt.prev = prev

    self.size -= 1

    return node

  def delete_tail(self):
    if self.tail is None:
      return None
    elif self.tail == self.head:
      val = self.tail.val
      self.tail = None
      self.head = None
      self.size = 0
      return val

    node = self.tail
    self.tail = node.prev
    self.tail.next = None

    self.size -= 1

    return node.val

  def prepend(self, val):
    if self.head is None and self.tail is None:
      self.head = ListNode(val, None, None)
      self.tail = self.head
    elif self.head is None:
      self.head = ListNode(val, self.tail, None)
      self.tail.prev = self.head
    else:
      self.head = ListNode(val, self.head, None)
      self.head.next.prev = self.head

    self.size += 1

    return self

  def append(self, val):
    if self.head is None and self.tail is None:
      self.head = ListNode(val, None, None)
      self.tail = self.head
    elif self.tail is None:
      self.tail = ListNode(val, None, self.head)
      self.head.next = self.tail
    else:
      self.tail = ListNode(val, None, self.tail)
      self.tail.prev.next = self.tail

    self.size += 1

    return self

  def __str__(self):
    return str([ n.val for n in self ])

class LRU(object):
  def __init__(self, size=100):
    self.size = size
    self.queue = DoublyLinkedList()
    self.hash = {}

  def __len__(self):
    return self.queue.size

  def resize(self, new_size):
    if new_size < 0:
      raise ValueError("The LRU limit must be a positive number. Got: " + str(new_size))

    if new_size == 0:
      self.queue = DoublyLinkedList()
      self.hash = {}
      return

    if new_size >= self.size:
      self.size = new_size
      return 

    while len(self.queue) > new_size:
      (key,val) = self.queue.delete_tail()
      del self.hash[key]

  def __contains__(self, key):
    return key in self.hash

  def __getitem__(self, key):
    if key not in self.hash:
      raise KeyError("{} not in cache.".format(key))

    node = self.hash[key]
    self.queue.promote_to_head(node)

    return node.val[1]

  def __setitem__(self, key, val):
    pair = (key,val)
    if key in self.hash:
      node = self.hash[key]
      node.val = pair
      self.queue.promote_to_head(node)
      return

    self.queue.prepend(pair)
    self.hash[key] = self.queue.head

    while len(self.queue) > self.size:
      (tkey,tval) = self.queue.delete_tail()
      del self.hash[tkey]      

  def __str__(self):
    return str(self.queue)



