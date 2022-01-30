

class Selecter:
    def __init__(self) -> None:
        pass

class Fitnes:
    def __init__(self) -> None:
        pass

class GA:
    def __init__(self, params, Selecter, Fitnes, do_crowding:bool) -> None:
        self. params = params
        self.Selecter = Selecter
        self.Fitnes = Fitnes
        self.do_crowding = do_crowding

    def init_pop(self):
        pass

    def evaluate_pop(self, pop):
        pass

    def do_terminate(self, pop_eval, gen_count):
        pass

    def select_parents(self, pop):
        pass

    def make_offsprings(self, parents):
        pass

    def select_survivors(self, old_pop, offsprings, pop_eval, offs_eval):
        pass

    def plot_progress(self):
        pass

    def plot_end_result(self):
        pass

    def run(self):
        pop = self.init_pop() # numpy array (pop_size, 1)
        gen_count = 0
        pop_eval = self.evaluate_pop(pop)
        eval_log = {gen_count: pop_eval}
        while not self.do_terminate(pop_eval, gen_count):
            parents = self.select_parents(pop)
            offsprings = self.make_offsprings(parents)
            offs_eval = self.evaluate_pop(offsprings) 
            pop = self.select_survivors(pop, offsprings, pop_eval, offs_eval)

            gen_count += 1
            pop_eval = self.evaluate_pop(pop)
            eval_log[gen_count] = pop_eval
        
        return pop, eval_log
