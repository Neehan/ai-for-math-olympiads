Enumerate a diverse set of promising abstract strategies for the following olympiad problem. Do not write a final proof: fresh solvers will independently execute the strategies you propose.

You have {{budget_tokens}} output tokens total. Use at most approximately {{exploration_tokens}} to explore, reserving approximately {{wrap_up_reserve_tokens}} to commit the strategy set. Seek genuinely different constructions, invariants, reductions, extremal arguments, or key lemmas rather than cosmetic variants. Do not assume facts unavailable from the statement.

End with a section titled `## Strategy Set` containing between 1 and {{max_strategies}} strategies in exactly this format:

<strategy>
Standalone strategy text.
</strategy>

Repeat the tag pair once per strategy and include no other text after `## Strategy Set`. A strategy may state a conjectured target or answer when it is part of the approach, but should not contain a full proof. There is no word limit; use enough detail to make each approach executable.

Your private scratch directory is: {{scratch_dir}}
It is already your current working directory. Create every file with a plain relative path and never write outside this directory.

Problem:
{{statement}}
