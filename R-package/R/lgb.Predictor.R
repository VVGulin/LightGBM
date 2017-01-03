Predictor <- R6Class(
  "lgb.Predictor",
  public = list(
    handle = NULL,
    initialize = function(modelfile) {
      if(typeof(modelfile) == "character") {
        handle <- .Call("LGBM_BoosterCreateFromModelfile_R", modelfile, PACKAGE="lightgbm")
      } else if (typeof(modelfile) == "lgb.Booster.handle") {
        handle <- modelfile
      } else {
        stop("modelfile must be either character filename, or lgb.Booster.handle")
      }
      class(handle) <- "lgb.Booster.handle"
      self$handle <- handle
    },

    predict = function(data, 
      num_iteration = NULL, rawscore = FALSE, predleaf = FALSE, header = FALSE, 
      reshape = FALSE, ...) {

      if (is.null(num_iteration)) {
        num_iteration <- -1
      }

      num_row <- 0
      if (typeof(data) == "character") {
        ret <- .Call("LGBM_BoosterPredictForMat_R", self$handle, data, as.integer(header),
          as.integer(rawscore),
          as.integer(predleaf),
          as.integer(num_iteration),
          num_row,
          PACKAGE = "lightgbm")
      } else if (is.matrix(data)) {
        ret <- .Call("LGBM_BoosterPredictForMat_R", self$handle, data,
          as.integer(rawscore),
          as.integer(predleaf),
          as.integer(num_iteration),
          PACKAGE = "lightgbm")
        num_row <- nrow(data)
      } else if (class(data) == "dgCMatrix") {
        ret <- .Call("LGBM_BoosterPredictForCSC_R", self$handle, data@p, data@i, data@x, nrow(data),
          as.integer(rawscore),
          as.integer(predleaf),
          as.integer(num_iteration),
          PACKAGE = "lightgbm")
        num_row <- nrow(data)
      } else {
        stop(paste("predict.lgb.Predictor: does not support to predict from ",
                   typeof(data)))
      }

      if (length(ret) %% num_row != 0) {
        stop("prediction length ", length(ret)," is not multiple of nrows(data) ", num_row)
      }
      npred_per_case <- length(ret) / num_row
      if (reshape && npred_per_case > 1) {
        ret <- matrix(ret, ncol = npred_per_case)
      }
      return(ret)
    }
  )
)
