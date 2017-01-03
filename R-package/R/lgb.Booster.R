Booster <- R6Class(
  "lgb.Booster",
  public = list(
    handle = NULL,
    best_iter = -1,
    initialize = function(params = list(),
                          train_set = NULL,
                          modelfile = NULL,
                          ...) {
      params <- append(params, list(...))
      params_str <- as.character(lgb.params2str(params))
      if (!is.null(train_set)) {
        if (!lgb.check.r6.class(train_set, "lgb.Dataset")) {
          stop("lgb.Booster: Only can use lgb.Dataset as training data")
        }
        self$handle <-
          .Call("LGBM_BoosterCreate_R", train_set, params_str, PACKAGE = "lightgbm")
        private$train_set <- train_set
        private$num_dataset <- 1
        private$init_predictor <- train_set$predictor
        if (!is.null(private$init_predictor)) {
          .Call("LGBM_BoosterMerge_R",
                self$handle,
                private$init_predictor$handle,
                PACKAGE = "lightgbm")
        }
        private$predict_buffer <- c(private$predict_buffer, NULL)
        private$is_predicted_cur_iter <-
          c(private$is_predicted_cur_iter, FALSE)
      } else if (!is.null(modelfile)) {
        if (!is.character(modelfile)) {
          stop("lgb.Booster: Only can use string as model file path")
          self$handle <-
            .Call("LGBM_BoosterCreateFromModelfile_R",
                  modelfile,
                  PACKAGE = "lightgbm")
        }
      } else {
        stop(
          "lgb.Booster: Need at least one training dataset or model file to create booster instance"
        )
      }
      class(self$handle) <- "lgb.Booster.handle"
      private$num_class <-
        .Call("LGBM_BoosterGetNumClasses_R", self$handle, PACKAGE = "lightgbm")
    },
    set_train_data_name = function(name) {
      private$name_train_set <- name
      return(self)
    },
    add_valid = function(data, name) {
      if (!lgb.check.r6.class(data, "lgb.Dataset")) {
        stop("lgb.Booster.add_valid: Only can use lgb.Dataset as validation data")
      }
      if (!identical(data$predictor, private$init_predictor)) {
        stop(
          "lgb.Booster.add_valid: Add validation data failed, you should use same predictor for these data"
        )
      }
      .Call("LGBM_BoosterAddValidData_R", self$handle, data, PACKAGE = "lightgbm")
      private$valid_sets <- c(private$valid_sets, data)
      private$name_valid_sets <- c(private$name_valid_sets, name)
      private$num_dataset <- private$num_dataset + 1
      private$predict_buffer <- c(private$predict_buffer, NULL)
      private$is_predicted_cur_iter <-
        c(private$is_predicted_cur_iter, FALSE)
      return(self)
    },
    reset_parameter = function(params, ...) {
      params <- append(params, list(...))
      params_str <- as.character(lgb.params2str(params))
      .Call("LGBM_BoosterResetParameter_R",
            self$handle,
            params_str,
            PACKAGE = "lightgbm")
      return(self)
    },
    update = function(train_set = NULL, fobj = NULL) {
      if (!is.null(train_set)) {
        if (!lgb.check.r6.class(train_set, "lgb.Dataset")) {
          stop("lgb.Booster.update: Only can use lgb.Dataset as training data")
        }
        if (!identical(train_set$predictor, private$init_predictor)) {
          stop(
            "lgb.Booster.update: Change train_set failed, you should use same predictor for these data"
          )
        }
        .Call("LGBM_BoosterResetTrainingData_R",
              self$handle,
              train_set,
              PACKAGE = "lightgbm")
        private$train_set = train_set
        private$predict_buffer[1] <- NULL
      }
      if (is.null(fobj)) {
        ret <-
          .Call("LGBM_BoosterUpdateOneIter_R", self$handle, PACKAGE = "lightgbm")
      } else {
        if (typeof(fobj) != 'closure') {
          stop("lgb.Booster.update: fobj should be a function")
        }
        gpair <- fobj(private$inner_predict(1), private$train_set)
        ret <-
          .Call(
            "LGBM_BoosterUpdateOneIterCustom_R",
            self$handle,
            gpair$grad,
            gpair$hess,
            PACKAGE = "lightgbm"
          )
      }
      for (i in 1:length(private$is_predicted_cur_iter)) {
        private$is_predicted_cur_iter[i] <- FALSE
      }
      return(ret)
    },
    rollback_one_iter = function() {
      .Call("LGBM_BoosterRollbackOneIter_R", self$handle, PACKAGE = "lightgbm")
      for (i in 1:length(private$is_predicted_cur_iter)) {
        private$is_predicted_cur_iter[i] <- FALSE
      }
      return(self)
    },
    current_iter = function() {
      return(.Call("LGBM_BoosterGetCurrentIteration_R", self$handle, PACKAGE = "lightgbm"))
    },
    eval = function(data, name, feval = NULL) {
      if (!lgb.check.r6.class(data, "lgb.Dataset")) {
        stop("lgb.Booster.eval: only can use lgb.Dataset to eval")
      }
      data_idx <- 0
      if (identical(data, private$train_set)) {
        data_idx <- 1
      } else {
        for (i in 1:length(private$valid_sets)) {
          if (identical(data, private$valid_sets[i])) {
            data_idx <- i + 1
            break
          }
        }
      }
      if (data_idx == 0) {
        self$add_valid(data, name)
        data_idx <- private$num_dataset
      }
      return(private$inner_eval(name, data_idx, feval))
    },
    eval_train = function(feval = NULL) {
      return(private$inner_eval(private$name_train_set, 1, feval))
    },
    eval_valid = function(feval = NULL) {
      ret = list()
      for (i in 1:length(private$valid_sets)) {
        ret <-
          c(ret,
            private$inner_eval(private$name_valid_sets[i], i + 1, feval))
      }
      return(ret)
    },
    save_model = function(filename, num_iteration = NULL) {
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }
      .Call(
        "LGBM_BoosterSaveModel_R",
        self$handle,
        num_iteration,
        as.character(filename),
        PACKAGE = "lightgbm"
      )
      return(self)
    },
    dump_model = function(num_iteration = NULL) {
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }
      return(
        .Call(
          "LGBM_BoosterDumpModel_R",
          self$handle,
          num_iteration,
          PACKAGE = "lightgbm"
        )
      )
    },
    predict = function(data,
                        num_iteration = NULL,
                        rawscore = FALSE,
                        predleaf = FALSE,
                        header = FALSE,
                        reshape = FALSE) {
      if (is.null(num_iteration)) {
        num_iteration <- self$best_iter
      }
      predictor <- Predictor$new(self$handle)
      return(predictor$predict(data, num_iteration, rawscore, predleaf, header, reshape))
    },
    to_predictor = function() {
      Predictor$new(self$handle)
    }
  ),
  private = list(
    train_set = NULL,
    name_train_set = "training",
    valid_sets = list(),
    name_valid_sets = list(),
    predict_buffer = list(),
    is_predicted_cur_iter = list(),
    num_class = 1,
    num_dataset = 0,
    init_predictor = NULL,
    eval_names = NULL,
    higher_better_inner_eval = NULL,
    inner_predict = function(idx) {
      if (idx > private$num_dataset) {
        stop("data_idx should not be greater than num_dataset")
      }
      if (is.null(private$predict_buffer[idx])) {
        npred <-
          .Call("LGBM_BoosterGetNumPredict_R",
                self$handle,
                idx - 1,
                PACKAGE = "lightgbm")
        private$predict_buffer[idx] <- rep(0.0, npred)
      }
      if (!private$is_predicted_cur_iter[idx]) {
        .Call(
          "LGBM_BoosterGetNumPredict_R",
          self$handle,
          idx - 1,
          private$predict_buffer[idx],
          PACKAGE = "lightgbm"
        )
        private$is_predicted_cur_iter[idx] <- TRUE
      }
      return(private$predict_buffer[idx])
    },
    get_eval_info = function() {
      if (is.null(private$eval_names)) {
        private$eval_names <-
          .Call("LGBM_BoosterGetEvalNames_R", self$handle, PACKAGE = "lightgbm")
        if (!is.null(private$eval_names)) {
          private$higher_better_inner_eval <-
            rep(FALSE, length(private$eval_names))
          for (i in 1:length(private$eval_names)) {
            if (startsWith(private$eval_names[i], "auc")
                | startsWith(private$eval_names[i], "ndcg")) {
              private$higher_better_inner_eval[i] <- TRUE
            }
          }
        }
      }
      if (is.null(private$eval_names)) {
        private$eval_names <- list()
      }
      return(private$eval_names)
    },
    inner_eval = function(data_name, data_idx, feval = NULL) {
      if (idx > private$num_dataset) {
        stop("data_idx should not be greater than num_dataset")
      }
      private$get_eval_info()
      ret <- list()
      if (length(private$eval_names) > 0) {
        tmp_res <-
          .Call("LGBM_BoosterGetEval_R",
                self$handle,
                data_idx - 1,
                PACKAGE = "lightgbm")
        for (i in 1:length(tmp_res)) {
          ret <-
            c(
              ret,
              c(
                data_name,
                private$eval_names[i],
                tmp_res[i],
                private$higher_better_inner_eval[i]
              )
            )
        }
      }
      if (!is.null(feval)) {
        if (typeof(feval) != 'closure') {
          stop("lgb.Booster.eval: feval should be a function")
        }
        data <- private$train_set
        if (data_idx > 1) {
          data <- private$valid_sets[data_idx - 1]
        }
        res <- feval(private$inner_predict(data_idx), data)
        for (i in 1:length(res)) {
          ret <-
            c(ret,
              c(data_name, res[i]$name, res[i]$value, res[i]$higher_better))
        }
      }
      return(ret)
    }
  )
)

#' Predict method for LightGBM model
#' 
#' Predicted values based on either lightgbm model or model handle object.
#' 
#' @param booster Object of class \code{lgb.Booster}
#' @param data a \code{matrix} object, a \code{dgCMatrix} object or a character representing a filename
#' @param num_iteration number of iteration want to predict with, <= 0 means use best iteration
#' @param rawscore whether the prediction should be returned in the for of original untransformed 
#'        sum of predictions from boosting iterations' results. E.g., setting \code{rawscore=TRUE} for 
#'        logistic regression would result in predictions for log-odds instead of probabilities.
#' @param predleaf whether predict leaf index instead. 
#' @param header only used for prediction for text file. True if text file has header
#' @param reshape whether to reshape the vector of predictions to a matrix form when there are several 
#'        prediction outputs per case.
#' 
#' @details  
#' Note that \code{ntreelimit} is not necessarily equal to the number of boosting iterations
#' and it is not necessarily equal to the number of trees in a model.
#' E.g., in a random forest-like model, \code{ntreelimit} would limit the number of trees.
#' But for multiclass classification, there are multiple trees per iteration, 
#' but \code{ntreelimit} limits the number of boosting iterations.
#' 
#' Also note that \code{ntreelimit} would currently do nothing for predictions from gblinear, 
#' since gblinear doesn't keep its boosting history. 
#' 
#' One possible practical applications of the \code{predleaf} option is to use the model 
#' as a generator of new features which capture non-linearity and interactions, 
#' e.g., as implemented in \code{\link{xgb.create.features}}. 
#' 
#' @return 
#' For regression or binary classification, it returns a vector of length \code{nrows(data)}.
#' For multiclass classification, either a \code{num_class * nrows(data)} vector or 
#' a \code{(nrows(data), num_class)} dimension matrix is returned, depending on 
#' the \code{reshape} value.
#' 
#' When \code{predleaf = TRUE}, the output is a matrix object with the 
#' number of columns corresponding to the number of trees.
#' 
#'
#' @rdname predict.lgb.Booster
#' @export
predict.lgb.Booster <- function(booster, 
                        data,
                        num_iteration = NULL,
                        rawscore = FALSE,
                        predleaf = FALSE,
                        header = FALSE,
                        reshape = FALSE) {
  booster$predict(data, num_iteration, rawscore, predleaf, header, reshape)
}

#' Save LightGBM model
#' 
#' Save LightGBM model
#' 
#' @param booster Object of class \code{lgb.Booster}
#' @param filename saved filename
#' @param num_iteration number of iteration want to predict with, <= 0 means use best iteration
#' 
#' @return booster
#' 
#'
#' @rdname lgb.save 
#' @export
lgb.save <- function(booster, filename, num_iteration=NULL){
  booster$save_model(booster, filename, num_iteration)
}

#' Dump LightGBM model to json
#' 
#' Dump LightGBM model to json
#' 
#' @param booster Object of class \code{lgb.Booster}
#' @param num_iteration number of iteration want to predict with, <= 0 means use best iteration
#' 
#' @return json format of model
#' 
#'
#' @rdname lgb.dump 
#' @export
lgb.dump <- function(booster, num_iteration=NULL){
  booster$dump_model(booster, num_iteration)
}