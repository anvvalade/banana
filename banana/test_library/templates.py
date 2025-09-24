from banana import (
    BaseVariable,
    BaseModule,
    BaseUtil,
    BaseAnalysis,
    i_cast,
    f_cast,
    DataType,
    ScalarType,
)

from tensorflow_probability.substrates.jax import distributions as tfd


class MyUtil(BaseUtil):

    def __init__(
        self,
        fixed_param_1: float,
        fixed_param_2: ScalarType,
        obs_data_1: DataType,
        obs_data_2: DataType,
        verbosity: int = 2,
    ):
        super().__init__(verbosity)

    def whateverYouWant(self, what_ever_you_want):
        return what_ever_you_want


class MyVariable(BaseVariable):
    provides = "a_free_parameter"

    def __init__(
        self,
        fixed_param_1: float,
        fixed_param_2: ScalarType,
        obs_data_1: DataType,
        obs_data_2: DataType,
        MU: MyUtil,
        verbosity: int = 2,
    ):
        super().__init__(verbosity)
        self.output_size = 1
        self.setPriorMeanStdAndMM(0.0, 1.0, dmm=1)
        self.init_state_gen = tfd.Normal(0.0, 1.0)


class MyModule(BaseModule):
    provides = "a_physical_quantity"

    def __init__(
        self,
        fixed_param_1: float,
        fixed_param_2: ScalarType,
        obs_data_1: DataType,
        obs_data_2: DataType,
        MU: MyUtil,
        verbosity: int = 2,
    ):
        super().__init__(verbosity)

        self.float32_jax_array = f_cast(obs_data_1)
        self.int32_jax_array = i_cast(fixed_param_1)

    def call(self, a_free_parameter, another_physical_quantity):
        # no numpy, no matplotlib, just jax here!
        return a_free_parameter * another_physical_quantity / self.float32_jax_array

    def debug(self, a_free_parameter, another_physical_quantity):
        # use numpy, matplotlib, anything here -> but slow!
        return a_free_parameter * another_physical_quantity / self.float32_jax_array

    def details(self, a_free_parameter, another_physical_quantity):
        # no numpy, no matplotlib, just jax here!
        return dict(
            a_physical_quantity=a_free_parameter
            * another_physical_quantity
            / self.float32_jax_array,
            intermediate_quantity=a_free_parameter * another_physical_quantity,
        )


class MyAnalysis(BaseAnalysis):

    def __init__(
        self,
        fixed_param_1: float,
        fixed_param_2: ScalarType,
        obs_data_1: DataType,
        obs_data_2: DataType,
        MU: MyUtil,
        verbosity: int = 2,
    ):
        super().__init__(verbosity)

    def oneState(self, real, any_quantity_from_details):
        # numpy, matplotlib
        # save, plot, anything you want
        # arguments here are ONE SINGLE Monte Carlo state, not the whole chain
        # this saves memory
        # real is the realization
        # the total number of realizations is not known in advance (could change...)
        return

    def finalize(self, any_other_quantity_from_details_or_no_arg_at_all):
        # numpy, matplotlib
        # save, plot, anything you want
        # arguments here are THE WHOLE CHAIN
        return
