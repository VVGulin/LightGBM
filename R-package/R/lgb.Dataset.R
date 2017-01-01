Dataset <- R6Class(
  "lgb.Dataset",
  public = list(
    handle = NULL,
    raw_data = NULL,
    params = NULL,
    reference = NULL,
    colnames = NULL,
    categorical_feature = NULL,
    predictor = NULL,
    free_raw_data = TRUE,
    used_indices = NULL,
    info = NULL,
    initialize = function(data,
                          params = list(),
                          reference = NULL,
                          colnames = NULL,
                          categorical_feature = NULL,
                          predictor = NULL,
                          free_raw_data = TRUE,
                          used_indices = NULL,
                          info = list(),
                          ...) {
      info <- append(info, list(...))
      if (!is.null(reference)) {
        if (!lgb.check.r6.class(reference, "lgb.Dataset")) {
          stop("Only can use lgb.Dataset as reference")
        }
      }
      if (!is.null(predictor)) {
        if (!lgb.check.r6.class(predictor, "lgb.Predictor")) {
          stop("Only can use lgb.Predictor as predictor")
        }
      }
      self$raw_data <- data
      self$params <- params
      self$reference <- reference
      self$colnames <- colnames
      
      self$categorical_feature <- categorical_feature
      self$predictor <- predictor
      self$free_raw_data <- free_raw_data
      self$used_indices <- used_indices
      self$info <- info
    },
    create_valid = function(data, info = list(),  ...) {
      info <- append(info, list(...))
      ret <- Dataset$new(
        data,
        self$params,
        self,
        self$colnames,
        self$categorical_feature,
        self$predictor,
        self$free_raw_data,
        NULL,
        info
      )
      return(ret)
    },
    get_handle = function() {
      if (is.null(self$handle)) {
        self$construct()
      }
      return(self$handle)
    },
    construct = function() {
      if (!is.null(self$handle)) {
        return(self)
      }
      # Get feature names
      cnames <- NULL
      if (is.matrix(self$raw_data) |
          class(self$raw_data) == "dgCMatrix") {
        cnames <- colnames(self$raw_data)
      }
      # set feature names if not exist
      if (is.null(self$colnames)) {
        self$colnames <- as.list(cnames)
      }
      # Get categorical feature index
      if (!is.null(self$categorical_feature)) {
        fname_dict <- list()
        if (!is.null(self$colnames)) {
          fname_dict <-
            as.list(setNames(0:(length(
              self$colnames
            ) - 1), self$colnames))
        }
        cate_indices <- list()
        for (key in self$categorical_feature) {
          if (is.character(key)) {
            idx <- fname_dict[[key]]
            if (is.null(idx)) {
              stop(paste("lgb.self.get.handle: cannot find feature name ", key))
            }
            cate_indices <- append(cate_indices, idx)
          } else {
            # one-based indices to zero-based
            idx <- as.integer(key - 1)
            cate_indices <- append(cate_indices, idx)
          }
        }
        self$params$categorical_feature <- cate_indices
      }
      # Check has header or not
      has_header <- FALSE
      if (!is.null(self$params$has_header) |
          !is.null(self$params$header)) {
        if (tolower(as.character(self$params$has_header)) == "true"
            | tolower(as.character(self$params$header)) == "true") {
          has_header <- TRUE
        }
      }
      # Generate parameter str
      params_str <- as.character(lgb.params2str(self$params))
      # get handle of reference dataset
      ref_handle <- NULL
      if (!is.null(self$reference)) {
        ref_handle <- self$reference$get_handle()
      }
      # not subset
      if (is.null(self$used_indices)) {
        if (typeof(self$raw_data) == "character") {
          handle <-
            .Call(
              "LGBM_DatasetCreateFromFile_R",
              self$raw_data,
              params_str,
              ref_handle,
              PACKAGE = "lightgbm"
            )
        } else if (is.matrix(self$raw_data)) {
          handle <-
            .Call(
              "LGBM_DatasetCreateFromMat_R",
              self$raw_data,
              params_str,
              ref_handle,
              PACKAGE = "lightgbm"
            )
        } else if (class(self$raw_data) == "dgCMatrix") {
          handle <- .Call(
            "LGBM_DatasetCreateFromCSC_R",
            self$raw_data@p,
            self$raw_data@i,
            self$raw_data@x,
            nrow(self$raw_data),
            params_str,
            ref_handle,
            PACKAGE = "lightgbm"
          )
        } else {
          stop(paste(
            "lgb.Dataset.construct: does not support to construct from ",
            typeof(self$raw_data)
          ))
        }
      } else {
        # construct subset
        if (is.null(self$reference)) {
          stop("lgb.Dataset.construct: reference cannot be NULL if construct subset")
        }
        handle <-
          .Call("LGBM_DatasetGetSubset_R",
                ref_handle,
                self$used_indices,
                params_str,
                PACKAGE = "lightgbm")
      }
      
      class(handle) <- "lgb.Dataset.handle"
      self$handle <- handle
      # set feature names
      if (!is.null(self$colnames)) {
        .Call("LGBM_DatasetSetFeatureNames_R",
              self$handle,
              self$colnames,
              PACKAGE = "lightgbm")
      }
      
      # load init score
      if (!is.null(self$predictor) & is.null(self$used_indices)) {
        init_score <-
          self$predictor$predict(self$raw_data, rawscore = TRUE, reshape = TRUE)
        # not need to transpose, for is col_marjor
        init_score <- as.vector(init_score)
        self$info$init_score <- init_score
      }
      
      if (self$free_raw_data) {
        self$raw_data <- NULL
      }
      
      if (length(self$info) > 0) {
        # set infos
        for (i in 1:length(self$info)) {
          p <- self$info[i]
          self$setinfo(names(p), p[[1]])
        }
      }
      
      if (is.null(self$getinfo("label"))) {
        stop("lgb.Dataset.construct: label should be set")
      }
      return(self)
    },
    dim = function() {
      if (!is.null(self$handle)) {
        return(c(
          .Call("LGBM_DatasetGetNumData_R", self$handle, PACKAGE = "lightgbm"),
          .Call("LGBM_DatasetGetNumFeature_R", self$handle, PACKAGE = "lightgbm")
        ))
      } else if (is.matrix(self$raw_data) |
                 class(self$raw_data) == "dgCMatrix") {
        return(dim(self$raw_data))
      } else {
        stop(
          "cannot get Dimensions before dataset constructed, please call construct.lgb.Dataset explicit"
        )
      }
    },
    set_colnames = function(colnames) {
      self$colnames <- as.list(colnames)
      if (!is.null(self$colnames) & !is.null(self$handle)) {
        .Call("LGBM_DatasetSetFeatureNames_R",
              self$handle,
              self$colnames,
              PACKAGE = "lightgbm")
      }
    },
    getinfo = function(name) {
      if (typeof(name) != "character" ||
          length(name) != 1 ||
          !name %in% c('label', 'weight', 'init_score', 'group')) {
        stop(
          "getinfo: name must one of the following\n",
          "    'label', 'weight', 'init_score', 'group'"
        )
      }
      if (is.null(self$info$name) & !is.null(self$handle)) {
        ret <-
          .Call("LGBM_DatasetGetField_R", self$handle, name, PACKAGE = "lightgbm")
        if (length(ret) > 0) {
          self$info$name <- ret
        }
      }
      return(self$info$name)
    },
    setinfo = function(name, info) {
      if (typeof(name) != "character" ||
          length(name) != 1 ||
          !name %in% c('label', 'weight', 'init_score', 'group')) {
        stop(
          "setinfo: name must one of the following\n",
          "    'label', 'weight', 'init_score', 'group'"
        )
        return(FALSE)
      }
      if (name == "group") {
        info <- as.integer(info)
      } else {
        info <- as.numeric(info)
      }
      self$info$name <- info
      if (!is.null(self$handle)) {
        .Call("LGBM_DatasetSetField_R",
              self$handle,
              name,
              self$info$name,
              PACKAGE = "lightgbm")
      }
      return(TRUE)
    },
    slice = function(idxset) {
      ret <- Dataset$new(
        NULL,
        self$params,
        self,
        self$colnames,
        self$categorical_feature,
        self$predictor,
        self$free_raw_data,
        idxset,
        NULL
      )
      return(ret)
    },
    set_categorical_feature = function(categorical_feature) {
      if (is.null(self$raw_data)) {
        stop(
          "cannot set categorical feature after free raw data,
          please set free_raw_data=FALSE when construct lgb.Dataset"
        )
      }
      self$categorical_feature <- categorical_feature
      self$handle <- NULL
    },
    set_predictor = function(predictor) {
      if (is.null(self$raw_data)) {
        stop(
          "cannot set predictor after free raw data,
          please set free_raw_data=FALSE when construct lgb.Dataset"
        )
      }
      if (!is.null(predictor)) {
        if (!lgb.check.r6.class(predictor, "lgb.Predictor")) {
          stop("Only can use lgb.Predictor as predictor")
        }
      }
      self$predictor <- predictor
      self$handle <- NULL
    },
    set_reference = function(reference) {
      if (is.null(self$raw_data)) {
        stop(
          "cannot set reference after free raw data,
          please set free_raw_data=FALSE when construct lgb.Dataset"
        )
      }
      if (!is.null(reference)) {
        if (!lgb.check.r6.class(reference, "lgb.Dataset")) {
          stop("Only can use lgb.Dataset as reference")
        }
      }
      self$reference <- reference
      self$handle <- NULL
    },
    save_binary = function(fname) {
      .Call("LGBM_DatasetSaveBinary_R", self$handle, fname, PACKAGE = "lightgbm")
    }
  )
)

#' Contruct lgb.Dataset object
#'
#' Contruct lgb.Dataset object from dense matrix, sparse matrix
#' or local file (that was created previously by saving an \code{lgb.Dataset}).
#'
#' @param data a \code{matrix} object, a \code{dgCMatrix} object or a character representing a filename
#' @param params a list of parameters
#' @param reference reference dataset
#' @param categorical_feature categorical features
#' @param predictor initial predictor
#' @param free_raw_data TRUE for need to free raw data after construct
#' @param info a list of information of the lgb.Dataset object
#' @param ... other information to pass to \code{info}.
#'
#' @examples
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#' lgb.Dataset.save(dtrain, 'lgb.Dataset.data')
#' dtrain <- lgb.Dataset('lgb.Dataset.data')
#' @export
lgb.Dataset <- function(data,
                        params = list(),
                        reference = NULL,
                        colnames = NULL,
                        categorical_feature = NULL,
                        predictor = NULL,
                        free_raw_data = TRUE,
                        used_indices = NULL,
                        info = list(),
                        ...) {
  Dataset$new(
    data,
    params,
    reference,
    colnames,
    categorical_feature,
    predictor,
    free_raw_data,
    used_indices,
    info,
    ...
  )
}

#' Contruct a validation data according to training data
#' @param dataset lgb.Dataset object, training data
#' @param data a \code{matrix} object, a \code{dgCMatrix} object or a character representing a filename
#' @param info a list of information of the lgb.Dataset object
#' @param ... other information to pass to \code{info}.
#'
#' @examples
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#' lgb.Dataset.save(dtrain, 'lgb.Dataset.data')
#' dtrain <- lgb.Dataset('lgb.Dataset.data')
#' @export
create.valid.lgb.Dataset <-
  function(dataset, data, info = list(),  ...) {
    dataset$create_valid(data, info, ...)
  }

#' Construct Dataset explicit
#'
#' Returns NULL
#' @param dataset Object of class \code{lgb.Dataset}
#'
#' @export
construct.lgb.Dataset <- function(dataset) {
  dataset$construct()
}

#' Dimensions of lgb.Dataset
#'
#' Returns a vector of numbers of rows and of columns in an \code{lgb.Dataset}.
#' @param dataset Object of class \code{lgb.Dataset}
#'
#' @details
#' Note: since \code{nrow} and \code{ncol} internally use \code{dim}, they can also
#' be directly used with an \code{lgb.Dataset} object.
#'
#' @examples
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#'
#' stopifnot(nrow(dtrain) == nrow(train$data))
#' stopifnot(ncol(dtrain) == ncol(train$data))
#' stopifnot(all(dim(dtrain) == dim(train$data)))
#'
#' @export
dim.lgb.Dataset <- function(dataset) {
  dataset$dim()
}

#' Handling of column names of \code{lgb.Dataset}
#' @param dataset object of class \code{lgb.Dataset}
#' @rdname colnames.lgb.Dataset
#' @export
colnames.lgb.Dataset <- function(dataset) {
  dataset$colnames
}

#' Get a new Dataset containing the specified rows of
#' orginal lgb.Dataset object
#'
#' Get a new Dataset containing the specified rows of
#' orginal lgb.Dataset object
#'
#' @param object Object of class "lgb.Dataset"
#' @param idxset a integer vector of indices of rows needed
#' @param ... other parameters (currently not used)
#'
#' @examples
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#'
#' dsub <- slice(dtrain, 1:42)
#' labels1 <- getinfo(dsub, 'label')
#' dsub <- dtrain[1:42, ]
#' labels2 <- getinfo(dsub, 'label')
#' all.equal(labels1, labels2)
#'
#' @rdname slice.lgb.Dataset
#' @export
slice <- function(object, ...)
  UseMethod("slice")

#' @rdname slice.lgb.Dataset
#' @export
slice.lgb.Dataset <- function(object, idxset, ...) {
  object$slice(idxset)
}


#' Get information of an lgb.Dataset object
#'
#' Get information of an lgb.Dataset object
#' @param object Object of class \code{lgb.Dataset}
#' @param name the name of the information field to get (see details)
#' @param ... other parameters
#'
#' @details
#' The \code{name} field can be one of the following:
#'
#' \itemize{
#'     \item \code{label}: label lightgbm learn from ;
#'     \item \code{weight}: to do a weight rescale ;
#'     \item \code{group}: group size
#'     \item \code{init_score}: initial score is the base prediction lightgbm will boost from ;
#' }
#'
#' @examples
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#'
#' labels <- getinfo(dtrain, 'label')
#' setinfo(dtrain, 'label', 1-labels)
#'
#' labels2 <- getinfo(dtrain, 'label')
#' stopifnot(all(labels2 == 1-labels))
#' @rdname getinfo
#' @export
getinfo <- function(object, ...)
  UseMethod("getinfo")

#' @rdname getinfo
#' @export
getinfo.lgb.Dataset <- function(object, name, ...) {
  object$getinfo(name)
}

#' Set information of an lgb.Dataset object
#'
#' @param object Object of class "lgb.Dataset"
#' @param name the name of the field to get
#' @param info the specific field of information to set
#' @param ... other parameters
#'
#' @details
#' The \code{name} field can be one of the following:
#'
#' \itemize{
#'     \item \code{label}: label lightgbm learn from ;
#'     \item \code{weight}: to do a weight rescale ;
#'     \item \code{init_score}: initial score is the base prediction lightgbm will boost from ;
#'     \item \code{group}.
#' }
#'
#' @examples
#' data(agaricus.train, package='lightgbm')
#' train <- agaricus.train
#' dtrain <- lgb.Dataset(train$data, label=train$label)
#'
#' labels <- getinfo(dtrain, 'label')
#' setinfo(dtrain, 'label', 1-labels)
#' labels2 <- getinfo(dtrain, 'label')
#' stopifnot(all.equal(labels2, 1-labels))
#' @rdname setinfo
#' @export
setinfo <- function(object, ...)
  UseMethod("setinfo")

#' @rdname setinfo
#' @export
setinfo.lgb.Dataset <- function(object, name, info, ...) {
  object$setinfo(name, info)
}

#' set categorical feature of \code{lgb.Dataset}
#' @param dataset object of class \code{lgb.Dataset}
#' @param categorical_feature categorical features
#' @rdname set.categorical.feature.lgb.Dataset
#' @export
set.categorical.feature.lgb.Dataset <-
  function(dataset, categorical_feature) {
    dataset$set_categorical_feature(categorical_feature)
  }

#' set reference of \code{lgb.Dataset}
#' @param dataset object of class \code{lgb.Dataset}
#' @param reference object of class \code{lgb.Dataset}
#' @rdname set.reference.lgb.Dataset
#' @export
set.reference.lgb.Dataset <- function(dataset, reference) {
  dataset$set_reference(reference)
}

#' save \code{lgb.Dataset} to binary file
#' @param dataset object of class \code{lgb.Dataset}
#' @param fname object filename of output file
#' @rdname save.binary.lgb.Dataset
#' @export
save.binary.lgb.Dataset <- function(dataset, fname) {
  dataset$save_binary(fname)
}
