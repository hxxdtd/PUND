
class Attacker:
    def __init__(
                self,
                epoch,
                insertion_location,
                k,
                eval_seed,
                attack_type,
                no_adv,
                total_data,
                valid_seed,
                ):
        self.epoch = epoch
        self.insertion_location = insertion_location
        self.k = k
        self.eval_seed = eval_seed
        self.attack_type=attack_type
        self.no_adv=no_adv
        self.total_data = total_data
        self.valid_seed = valid_seed
        print(f"{'! '*3} attack_type is {self.attack_type}, no_adv is {self.no_adv}, valid_seed is {self.valid_seed}")
        