# Numeros
An expression evaluator, simplifier, and manipulator

Expression Demo
```
# use the parse function to create expressions from strings
e = parse("a + b")
print(e.get_value(a = 5, b = 2)) # will print `7`

# use Constant() and Variable() to create atomic expressions
print(Constant(8)) # will print `8`
print(Variable('x')) # will print`x`

# access arithmatic operators from `Ops` class
f = Ops.add.assign(Constant(2), Variable("x")) # this creates a copy of Ops.add with the correct inputs
print(f) # will print `2 + x`

# create EquationRules from expressions
rule = EquationRule(parse("2 + g"), parse("g * 7"))
# rule.apply will create a generator that yields all the new expressions
applied = next(rule.apply(f))
print(applied) # will print `x * 7`

# access arithmatic rules froms `Rules` class
prop = Rules.mul_communative # the communative property (`a * b` goes to `b * a`)
print(next(prop.apply(applied)) # will print `7 * x`

# apply multiple rules in all possible combinations with EquationRules.combinate
for expr, record in EquationRules.combinate(f, [rule, prop]):
  print(expr)
  # record keeps track of the rules used in the combinate function
  print(format_proof(record))

# simplify uses combinate and the standard rules to find the shortest equivelent expression
simplified, record = simplify(parse("a * (b - b)")
print(simplified) # `0`
print(format_proof(record))
# will print:
#  a * (b - b) | (1) Given                     
#  a * 0       | (2) Additive Inverse Property 
#  0           | (3) Zero Annihilative Property
```
