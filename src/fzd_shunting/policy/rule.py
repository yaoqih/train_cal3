class RulePolicy:
    def score(self, dispatcher, candidate):
        action = candidate.action
        if action.operation == "put":
            usefulness = 4 * candidate.destination_count - 2 * (
                action.count - candidate.destination_count
            )
        else:
            usefulness = 2 * (action.count - candidate.destination_count)
        return (
            20 * candidate.gain
            + 5 * candidate.suffix_gain
            + 2 * candidate.delivered_gain
            + usefulness
            - 0.02 * len(candidate.route)
        )

    def rank(self, dispatcher, candidates):
        return sorted(
            candidates,
            key=lambda c: (
                -self.score(dispatcher, c),
                c.action.line,
                c.action.operation,
                -c.action.count,
            ),
        )

    def choose(self, dispatcher, state, candidates, context=None):
        return self.rank(dispatcher, candidates)[0]
