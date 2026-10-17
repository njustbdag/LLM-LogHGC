class Template:
    def __init__(self, tokens):
        self.tokens = tokens
        self.freq = 1
        self.group_id = None


class CanonicalTemplate:
    def __init__(self, tokens):
        self.tokens = tokens


class Node:
    def __init__(self):
        self.children = {}
        self.wildcard = None
        self.templates = []


class TemplateTree:
    def __init__(self):
        self.root = Node()
        self.templates = []
        self.store = {}
        self._next_gid = 0

    def normalize(self, tokens):
        return tokens

    def insert(self, tokens):
        tokens = self.normalize(tokens)
        tpl = Template(tokens)
        self.templates.append(tpl)
        self._insert_trie(tokens, tpl)
        existing = self.find_compatible(tokens)
        if existing is None or existing.group_id is None:
            gid = self._new_group(tokens)
            tpl.group_id = gid
            return
        gid = existing.group_id
        tpl.group_id = gid
        existing.group_id = gid
        existing.freq += 1
        canon = self.store[gid]
        merged = self.merge(canon.tokens, tokens, gid=gid)
        if merged is not None:
            canon.tokens = merged

    def _insert_trie(self, tokens, tpl):
        node = self.root

        for t in tokens:
            if t == "<X>":
                if node.wildcard is None:
                    node.wildcard = Node()
                node = node.wildcard
            else:
                if t not in node.children:
                    node.children[t] = Node()
                node = node.children[t]

        node.templates.append(tpl)

    def same_structure(self, a, b):
        def is_placeholder(token):
            return isinstance(token, str) and token.startswith("<X>*")

        if len(a) != len(b):
            return False
        for x, y in zip(a, b):
            x_ph = is_placeholder(x)
            y_ph = is_placeholder(y)
            if x_ph != y_ph:
                return False
            if not x_ph and x != y:
                return False
        return True

    def find_compatible(self, tokens):
        key = self.structure_key(tokens)
        for tpl in self.templates:
            tpl_key = self.structure_key(tpl.tokens)
            if self.same_structure(key, tpl_key):
                # if tpl_key == key:
                return tpl
        return None

    def structure_key(self, tokens):
        key = []
        i = 0
        while i < len(tokens):
            if tokens[i] == "<X>":
                j = i
                while j < len(tokens) and tokens[j] == "<X>":
                    j += 1
                span_len = j - i
                key.append(f"<X>*{span_len}")
                i = j
            else:
                key.append(tokens[i])
                i += 1
        return key

    def _new_group(self, tokens):
        gid = self._next_gid
        self._next_gid += 1
        self.store[gid] = CanonicalTemplate(tokens)
        return gid

    def merge(self, a, b, gid=None, show_changes=False):
        a_, b_ = self.structure_key(a), self.structure_key(b)
        if not self.same_structure(a_, b_):
            return None
        res = []
        i = j = 0
        while i < len(a_) and j < len(b_):
            x = a_[i]
            y = b_[j]
            if x.startswith("<X>*") and y.startswith("<X>*"):
                n1 = int(x.split("*")[1])
                n2 = int(y.split("*")[1])
                n = min(n1, n2)
                res.extend(["<X>"] * n)
                i += 1
                j += 1
                continue
            if x == y:
                res.append(x)
                i += 1
                j += 1
        if show_changes:
            print("\n[MERGE GROUP UPDATE]", gid)
            print("OLD:", b)
            print("NEW:", res)
        return res

    def match(self, log):
        log = self.normalize(log)
        for tpl in self.templates:
            if self.match_template(tpl.tokens, log):
                gid = tpl.group_id
                return self.store[gid].tokens
                # return gid
        return None

    def match_template(self, tpl, log):
        if len(tpl) != len(log):
            return False

        for a, b in zip(tpl, log):
            if a == "<X>":
                continue
            if a != b:
                return False

        return True

    def show_store(self):
        for gid, canon in self.store.items():
            print(f"{gid}: {canon.tokens}")
