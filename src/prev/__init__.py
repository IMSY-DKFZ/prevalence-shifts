# slightly modify psrcal calibration routines
# a. remove error when calibration fails
# b. remove auto usage of bias if not requested (to properly model temp scaling)

import psrcal.calibration
import torch


# a. deactivate exception with non-converged optimization loop
def calibrate_wo_exception(trnscores, trnlabels, tstscores, calclass, quiet=True, **kwargs):
    obj = calclass(trnscores, trnlabels, **kwargs)

    paramvec, value, curve, success = psrcal.calibration.lbfgs(obj, 100, quiet=quiet)

    # this section has been removed
    # if not success:
    #     raise Exception("LBFGS was unable to converge")

    return obj.calibrate(tstscores), [obj.temp, obj.bias] if obj.has_bias else [obj.temp]


psrcal.calibration.calibrate = calibrate_wo_exception


# b. deactivate auto-pior usage as bias if not desired
def calibrate_wo_bias(self, scores):
    self.cal_scores = self.temp * scores + self.bias
    if self.priors is not None and self.has_bias:  # this condition has been added
        self.cal_scores += torch.log(self.priors)

    self.log_probs = self.cal_scores - torch.logsumexp(self.cal_scores, axis=-1, keepdim=True)
    return self.log_probs


psrcal.calibration.AffineCal.calibrate = calibrate_wo_bias
