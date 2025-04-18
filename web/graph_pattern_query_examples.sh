# Inductive reasoning - 4/30 files
python graph_pattern_query.py \
    --arity 1 \
    -q 'query(Z) :- node(X1, "example"), connected(X1, Y1), edge(Y1, Z, "reason:premise-conclusion"), node(X2, "example"), connected(X2, Y2), edge(Y2, Z, "reason:premise-conclusion"), X1 != X2, Y1 != Y2.' \

# Verification - 18/30 files
python graph_pattern_query.py \
    --arity 3 \
    -q 'query(X, Y, Z) :- edge(X, Y, "plan:frontier-verify"), node(Y, "planning"), connected(Y, Z), edge(X, Z, "evaluate:support").' \
    -q 'query(X, Y, Z) :- edge(X, Y, "plan:frontier-verify"), node(Y, "planning"), connected(Y, Z), edge(X, Z, "evaluate:refute").' \
    
# Backtracking - 11/30 files
python graph_pattern_query.py \
    --arity 2 \
    -q 'query(X, Y) :- edge(X, Y, "plan:plan-alternative"), node(X, "planning"), node(Y, "planning").' \
    
# Proof by contradiction - 4/30 files
python graph_pattern_query.py \
    --arity 2 \
    -q 'query(X, Y) :- node(X, "assumption"), node(Y, "reasoning"), connected(X, Y), edge(X, Y, "evaluate:refute").' \