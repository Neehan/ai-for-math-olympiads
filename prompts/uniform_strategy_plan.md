Identify all semantically distinct abstract solution strategies you can for the following olympiad problem, up to the stated maximum. Do not write a final proof: fresh solvers will independently execute the strategies you propose.

You have {{budget_tokens}} output tokens total. Use at most approximately {{exploration_tokens}} to explore, reserving approximately {{wrap_up_reserve_tokens}} for the final strategy set. This split is strict: stop exploratory reasoning and tool use before the exploration allowance is exhausted. Work at the conceptual level without writing the detailed proof. Seek genuinely different constructions, invariants, reductions, extremal arguments, or other load-bearing mechanisms rather than cosmetic variants. Do not assume facts unavailable from the statement.

Before emitting the final set, merge semantic duplicates and combine complementary components belonging to one proof. Every final entry must be a standalone plan for the entire problem: it must say how it would address every required direction, case, construction, or bound. Two entries are distinct only when their load-bearing proof mechanisms differ; a different lemma, representation, or presentation within the same mechanism is not enough. The entries are candidate strategies and need not be known correct, but each must describe an executable route to the complete result. Return fewer than {{max_strategies}} rather than pad the set with fragments or cosmetic duplicates.

End with a section titled `## Strategy Set` containing between 1 and {{max_strategies}} strategies in exactly this format:

<strategy>
Standalone strategy text.
</strategy>

Repeat the tag pair once per strategy and include no other text after `## Strategy Set`. Keep each strategy at most 500 words; return fewer strategies if necessary to stay within the reserved consolidation budget. A strategy may state a conjectured target or answer when it is part of the approach, but must identify it as conjectural and still cover the full proof obligation. Use enough conceptual detail to make each whole-proof plan executable without supplying the final proof itself.

Your private scratch directory is: {{scratch_dir}}
It is already your current working directory. Create every file with a plain relative path and never write outside this directory.

Problem:
{{statement}}
