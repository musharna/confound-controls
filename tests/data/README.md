# Reference data

`lalonde.csv` is the NSW / PSID comparison sample (Dehejia & Wahba's subset of
LaLonde's data) exactly as shipped in the R package MatchIt 4.5.5
(`data(lalonde)`), minus the `race` and `re78` columns.

`lalonde_matchit_logit.csv` and `lalonde_matchit_pairs.csv` are MatchIt's own
output on it, written by R and copied here verbatim:

    matchit(treat ~ age + educ + married + nodegree + re74 + re75, data = d,
            method = "nearest", distance = "glm", link = "linear.logit",
            caliper = 0.2, std.caliper = TRUE, m.order = "largest", replace = FALSE)

They are the external control for `tests/test_balance.py`: the numbers this
package computes are compared with what the reference implementation computed,
not with what this package computed last week.
