You are given four anonymous mathematical strategy sketches for the same problem. Exactly one is derived from a verified reference solution; the other three are derived from model-generated proposals and may or may not be valid. Rank all four from most to least likely to yield a complete correct proof of the problem. Evaluate the mathematical routes rather than prose polish, confidence, or presumed provenance. Do not write or complete a proof.

You have at most {{budget_tokens}} output tokens for this decision. You may use the provided offline scratch tools to test claims, seek counterexamples, and compare the strategies. Use at most {{working_tokens}} output tokens for this exploration; at that boundary, the harness stops exploration and reserves up to {{reserve_tokens}} output tokens for your final decision. Your submitted answer must remain only the requested ranking and concise reason.

Return exactly:

<ranking>one permutation of 1,2,3,4</ranking>
<reason>a concise reason for the ordering</reason>

Problem statement:

{{statement}}

Candidate strategies:

{{candidates}}
