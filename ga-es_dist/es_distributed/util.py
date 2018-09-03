
from .algos import ga, es, ga_moo


def get_algo_from_exp(exp):
    assert type(exp.policy.handler) is str, "Invalid policy type"
    if exp.policy.handler == 'ga_moo':
        return ga_moo.GAMoo()
    if exp.policy.handler == 'GA':
        return ga.GA()
    if exp.policy.handler == 'ES':
        return es.ES()
