import math, copy, pdb

def cache_result(method):
    def return_cache_if_present(self, *args, **kwargs):
        try:
            return self.__cache__[method]
        except AttributeError:
            method.__cache__ = { method : method(self, *args, **kwargs) }
            return method.__cache__[method]
    return return_cache_if_present

class Expression():
    def get_vars(self):
        """
        Return a list of variable names inside this expression or its children
        """
        return list(map(lambda var: var.name, filter(lambda inp: inp.__class__ == Variable, self.expand())))
    def get_conts(self):
        """
        Return a list of constant values inside this expression or its children
        """
        return list(map(lambda const: const.value, filter(lambda inp: inp.__class__ == Constant, self.expand())))
    def get_funcs(self):
        """
        Return a set of function display strings inside this expression and its children
        """
        return set(map(lambda fun: fun.disp, filter(lambda inp: inp.__class__ == Function, self.expand())))
    def insert_into(self, inserts):
        """
        Return the value at inserts[self], or a copy of self if it does not contain it
        """
        try:
            return inserts[self]
        except KeyError:
            return copy.deepcopy(self)
    def expand(self):
        """
        yield this expression and each child expression
        """
        yield self
    def walk(self, prefix = ""):
        """
        return a multiline string displaying this and all child expressions
        """
        return "{}{}: `{}`".format(prefix, self.__class__.__name__, self)
        
class Function(Expression):
    def __init__(self, func, disp, prec = None, inputs = None):
        """
        Create a Function expression
        func - the lambda to run in order to evaluate this expression
        disp - the format string to display. should include {0}, {1}, etc. for arguments
        prec - the precedence (if applicable). i.e. addition has a lower precedence then multiplication. used for placing parenthesis
        """
        self.func = func
        self.disp = disp
        self.prec = prec
        self.inputs = inputs
    def assign(self, *inputs):
        """
        Creates a copy with the inputs assigned
        returns the copy
        """
        clone = copy.deepcopy(self)
        clone.inputs = inputs
        return clone
    def __str__(self):
        if self.inputs == None:
            return self.disp
        else:
            return self.get_display(self.inputs)
    def walk(self, prefix = ""):
        """
        return a multiline string displaying this and all child expressions
        """
        s = prefix + self.disp
        for inp in self.inputs:
            s += prefix + "\n" + inp.walk(prefix = "   " + prefix)
        return s
    def expand(self):
        """
        yield this expression and each child expression
        """
        yield self
        for inp in self.inputs:
            for t in inp.expand():
                yield t
    def get_display(self, args, all_parens = False):
        """
        Use args to format this with its disp format string
        inserts parenthesis where needed
        """
        if self.prec != None:
            args = ["({})".format(inp) if all_parens or (hasattr(inp, "prec") and inp.prec != None and inp.prec < self.prec) else inp
                for inp in self.inputs]
        return self.disp.format(*args)
    def get_value(self, **bindings):
        """
        Evaluates this function
        bindings - map dictionary names to numeric values, used to evaluate variables
        returns evaluated result
        """
        return self.func(*[inp.get_value(**bindings) for inp in self.inputs])
    def substitute(self, **bindings):
        """
        Return a copy of this expression
        with the proper bindings substituted
        """
        return Function(self.func, self.disp, self.prec, inputs = [inp.substitute(**bindings) for inp in self.inputs])
    def get_bindings_in_children(self, expr):
        """
        Recursively yields the loc, bindings from self to expr and its children
            loc - Expression where `bindings` were found
            bindings - bindings from self to `loc`
        """
        parent_bindings = self.get_bindings(expr)
        if parent_bindings != None:
            yield expr, parent_bindings

        if hasattr(expr, 'inputs') and expr.inputs != None:
            for inp in expr.inputs:
                for loc, bindings in self.get_bindings_in_children(inp):
                    yield loc, bindings
        
    def get_bindings(self, expr):
        """
        Determine the bindings from `self` to `expr`. self.substitute(**bindings) will return a copy of expr
        For example:
          the binding from `a + 2` to `3 + 2` is {'a' : Constant(3)}
        returns the binding if it exists, None otherwise
        """
        
        if expr.__class__ == Function and expr.disp == self.disp and len(self.inputs) == len(expr.inputs):
            # compare children
            # line the inputs up, one by one
            # we must establish the match between each pair
            #pdb.set_trace()
            hasMatch = False
            compounded = {}
            for i in range(len(self.inputs)):
                binding = self.inputs[i].get_bindings(expr.inputs[i])
                if binding == None:
                    hasMatch = False
                    break
                hasMatch = True
                for key in binding:
                    if key in compounded and str(compounded[key]) != str(binding[key]):
                        hasMatch = False
                        break
                    else:
                        compounded[key] = binding[key] #TODO cleanup?
                if not hasMatch:
                    break
            if hasMatch:
                return compounded
            else:
                return None
        else:
            # the parents are not the same
            return None
        
    def insert_into(self, inserts):
        """
        Return the value at inserts[self], or a copy of this function where each child has been inserted into
        """
        if self in inserts:
            return inserts[self]
        else:
            return Function(self.func, self.disp, self.prec, inputs = [inp.insert_into(inserts) for inp in self.inputs])

class Constant(Expression):
    def __init__(self, num):
        """
        Creates an expression with a constant value
        num - value to hold
        """
        assert(num.__class__ == int or num.__class__ == float), "num must be a constant value"
        self.num = num
    def __str__(self):
        return str(self.num)
    def get_value(self, **bindings):
        """
        Evaluates this Constant value
        bindings - map dictionary names to numeric values, used to evaluate variables
        returns num value
        """
        return self.num
    def substitute(self, **bindings):
        """
        Return a copy of this expression
        with the proper bindings substituted
        """
        return Constant(self.num)
    def get_bindings(self, expr):
        """
        Determine the binding from `self` to `expr`. self.substitute(**bindings) will return a copy of expr
        For example:
          the binding from `a + 2` to `3 + 2` is {'a' : Constant(3)}
        returns the binding if it exists, None otherwise
        """
        if expr.__class__ == Constant and expr.num == self.num:
            return {}

class Variable(Expression):
    def __init__(self, name):
        """
        Creates an Variable expression
        name - name of variable
        """
        self.name = name
    def __str__(self):
        return str(self.name)
    def get_value(self, **bindings):
        """
        Evaluates this Variable
        bindings - map dictionary names to numeric values, used to evaluate variables
        returns the value at this variable's name in bindings,
        raises ValueError if it can't find the name inside bindings
        """
        try:
            return bindings[self.name]
        except KeyError:
            raise ValueError("bindings `{}` do not contain a binding for variable `{}` !".format(bindings, self.name))
    def get_bindings(self, expr):
        """
        Determine the binding from `self` to `expr`. self.substitute(**bindings) will return a copy of expr
        For example:
          the binding from `a + 2` to `3 + 2` is {'a' : Constant(3)}
        returns the binding if it exists, None otherwise
        """
        return {self.name : expr}
    def substitute(self, **bindings):
        """
        Return a copy of this expression
        with the proper bindings substituted
        """
        try:
            return bindings[self.name]
        except:
            return Variable(self.name)
        

class EquationRule():
    def __init__(self, pre, post, name = None):
        """
        Create an EquationRule
        pre - the expression before the rule is applied
        post - the expression after the rule is applied
        """
        self.vars = set(pre.get_vars())
        prevars = set(post.get_vars())
        assert(set(prevars).issubset(self.vars)), "Variables used in `post` ({}) must be subset of variables used in `pre` ({})".format(self.vars, prevars)
        self.pre = pre
        self.name = name
        self.post = post
    def __str__(self):
        return "{} => {}".format(self.pre, self.post)
    def get_name(self):
        return str(self) if self.name == None else self.name
    def create_inverse(self):
        return EquationRule(self.post, self.pre, name = "Inverse " + self.name)
    def apply(self, expr):
        """
        Apply this rule
        expr - the expression to apply it on
        return None if it couldn't apply the rule
        return a copy of expr with the rule applied
        """
        for loc, bindings in self.pre.get_bindings_in_children(expr): # CONSIDER~
            if bindings.keys() == self.vars:
                insert = self.post.substitute(**bindings)
                yield expr.insert_into({loc : insert})
    def combinate(orig_expr, rules):
        """
        Apply all the rules in all possible orders, breadth first
        expr - expression to apply rules to
        rules - iterable of EquationRules
        sofar - set of string forms of expressions
        that are not in `sofar`
        """
        layer = [(orig_expr, [(orig_expr, None)])] # list of (Expression, <reason>)
                                                   # <reason> is list of tuple of (Expression, EquationRule) to explain which steps lead to each
        sofar = set() # set of all string forms previously visited expressions
        to_str = lambda e: e.get_display(e.inputs, all_parens = True) if e.__class__ == Function else str(e) # so the set counts `(a + b) + c` differently from `a + (b + c)`
        
        while True:
            next_layer = []
            #pdb.set_trace()
            for expr, record in layer:
                if to_str(expr) not in sofar:
                    sofar.add(to_str(expr))
                    yield expr, record
                    
                    for rule in rules:
                        if rule.pre.get_funcs().issubset(expr.get_funcs()):
                            for new_expr in rule.apply(expr):
                                next_layer.append((new_expr, record + [(new_expr, rule)]))
            
            if len(next_layer) == 0:
                raise StopIteration()
            else:
                layer = next_layer
                

        



class Ops:
    add = Function(lambda x, y: x + y, "{0} + {1}", 10)
    sub = Function(lambda x, y: x - y, "{0} - {1}", 10) # interesting, subtraction must have higher precedence, or '3 - 2 + 1' is '3 - (2 + 1)'
    mul = Function(lambda x, y: x * y, "{0} * {1}", 20)
    div = Function(lambda x, y: x / y, "{0} / {1}", 20)
    exp = Function(lambda x, y: pow(x, y), "{0} ^ {1}", 30)
    # ALERT! update below whenever you update above
    infix_ops = {"^": exp, "*": mul, "/": div, "+": add, "-": sub}
    
    ln  = Function(lambda x: math.ln(x), "ln({0})")
    neg = Function(lambda x: math.ln(x), "-{0}")
    # ALERT! update below whenever you update above
    unary_ops = {"ln": ln, "-": neg}

def parse(string, infix_ops = Ops.infix_ops, unary_ops = Ops.unary_ops):
    def simple_builder(tokens, start, end):
        comped = tokens[:]
        ops = list(filter(lambda tok: tok.__class__ == Function and tok.inputs == None, comped)) # get all unfilled Ops
        try:
            upper_bound = max(map(lambda op: op.prec, filter(lambda op: op.prec != None, ops))) + 1 # the highest precedence we have available
        except ValueError:
            if len(comped) == 1:
                return comped[0]
            else:
                pdb.set_trace()
                raise ValueError("Can not evaluate multiple non-ops without operator (maximum precedence not defined): [{}]".format(", ".join(map(str, comped))))
        
        ops.sort(key = lambda op: upper_bound if op.prec == None else op.prec)
        ops.reverse()
        for op in ops:
            try:
                i = comped.index(op)
            except ValueError:
                pdb.set_trace()
            
            if i == 0 or i == len(comped) - 1:
                raise ValueError("Function `{}` without appropriate arguments\n\tcomped: [{}]\n\tops: [{}]".format(op, ", ".join(map(str, comped)), ", ".join(map(str, ops))))
            comped[i] = comped[i].assign(comped[i - 1], comped[i + 1])
            comped.pop(i+1)
            comped.pop(i-1)
            
        return comped[0]
    def paren_scanner(tokens, start):
        i = start
        exprs = []
        while i < len(tokens):
            token = tokens[i]
            if token == "(":
                e, new = paren_scanner(tokens, i+1)
                exprs.append(simple_builder(new, i, e))
                i = e
            elif token == ")":
                return i, exprs
            else:
                try:
                    exprs.append(infix_ops[token])
                except KeyError:
                    try:
                        exprs.append(unary_ops[token])
                    except KeyError:
                        try:
                            exprs.append(Constant(int(token)))
                        except ValueError:
                            exprs.append(Variable(token))
            i += 1

        return i, exprs
    def tokenize(string):
        tokens = []
        curr = ""
        for char in string:
            if char in infix_ops or char in unary_ops or char == '(' or char == ')':
                if curr != "": tokens.append(curr)
                curr = ""
                tokens.append(char)
            elif char != " ":
                curr += char
            else:
                if curr != "": tokens.append(curr)
                curr = ""
        if curr != "": tokens.append(curr)
        return tokens
    
    tokens = tokenize(string)
    i, nested_tokens = paren_scanner(tokens, 0)
    return simple_builder(nested_tokens, 0, i)

class Rules:    
    add_communative         = EquationRule(parse("a + b"),       parse("b + a"),name = "Communative Property of Addition")
    add_identity            = EquationRule(parse("a + 0"),       Variable("a"), name = "Additive Identity")
    add_associative         = EquationRule(parse("a + (b + c)"), parse("(a + b) + c"), name = "Associative Property of Addition")
    sub_inverse             = EquationRule(parse("a - a"),       Constant(0), name = "Additive Inverse Property")
    sub_to_mul_by_neg_one   = EquationRule(parse("a - b"),       Ops.add.assign(Variable("a"), Ops.mul.assign(Constant(-1), Variable("b"))), name = "Subtraction is Negative Addition")
    sub_identity            = EquationRule(parse("a - 0"),       Variable("a"), name = "Subtractive Identity")

    add_mul_distribute      = EquationRule(parse("a * (b + c)"), parse("a * b + a * c"), name = "Distributive Property of Addition")
    sub_mul_distribute      = EquationRule(parse("a * (b - c)"), parse("a * b - a * c"), name = "Distributive Property of Subtraction")

    mul_communative         = EquationRule(parse("a * b"),       parse("b * a"), name = "Communative Property of Multiplication")
    mul_identity            = EquationRule(parse("a * 1"),       Variable("a"), name = "Multiplicative Identity")
    mul_zero                = EquationRule(parse("a * 0"),       Constant(0), name = "Zero Annihilative Property")
    div_inverse             = EquationRule(parse("a / a"),       Constant(1), name = "Division Inverse Property")

    # WARNING! when you update above, update below as well
    
    rules =    [add_communative,
                add_identity,
                add_associative,
                sub_inverse,
                sub_to_mul_by_neg_one,
                sub_to_mul_by_neg_one.create_inverse(),
                sub_identity,
                add_mul_distribute,
                sub_mul_distribute,
                mul_communative,
                mul_identity,
                mul_zero,
                div_inverse]
    


def simplify(expr):
    simplest = expr
    simplest_record = [(expr, None)]
    shortest_length = len(list(expr.expand()))
    for other, record in EquationRule.combinate(expr, Rules.rules):
        length = len(list(other.expand()))
        if shortest_length == None or length < shortest_length:
            simplest = other
            simplest_record = record
            shortest_length = length
        elif length == shortest_length:
            ## ?
            pass
        
    return simplest, simplest_record

def format_proof(record):
    if len(record) == 0: return ""
    
    assert record[0][1] == None, "First step must be None as EquationRule"
    exprs, nums, reasons = [], [], []
    widths = (0, 0, 0)
    n = 1
    for expr, rule in record:
        exprs.append(str(expr))
        nums.append(str(n))
        n += 1
        reasons.append("Given" if rule == None else rule.name)

        recent = list(map(len, (exprs[-1], nums[-1], reasons[-1])))
        widths = ().__class__([max(widths[i], recent[i]) for i in range(3)])
    return "\n".join([
        "{0: <{width0}} | ({1: <{width1}}) {2: <{width2}}".format(exprs[i], nums[i], reasons[i], width0 = widths[0], width1 = widths[1], width2 = widths[2])
        for i in range(0, n - 1)
    ])
        
while True:
    expr = parse(input())
    simplified, record = simplify(expr)
    print()
    print(format_proof(record))
##    if input("enter debugger? [y/n] ") == "y":
##        pdb.set_trace()
    print()
    
    
