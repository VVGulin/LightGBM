#' Contruct lgb.Dataset object
#' 
#' Contruct lgb.Dataset object from dense matrix, sparse matrix 
#' or local file (that was created previously by saving an \code{lgb.Dataset}).
#' 
#' @param data a \code{matrix} object, a \code{dgCMatrix} object or a character representing a filename
#' @param params a list of parameters
#' @param reference refenence dataset
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
lgb.Dataset <- function(data, params=list(), reference=NULL, 
  categorical_feature=NULL, predictor=NULL, free_raw_data=TRUE, info=list(), ...) {
  info <- append(info, list(...))
  if(!is.null(reference)){
    if(!typeof(reference) == "lgb.Dataset"){
       stop("Only can use lgb.Dataset as reference")
    }
  }
  if(!is.null(predictor)){
    if(!typeof(predictor) == "lgb.Predictor"){
       stop("Only can use lgb.Predictor as predictor")
    }
  }
  ret <- structure(list(handle=NULL, raw_data=data,
    params=params, reference=reference, 
    categorical_feature=categorical_feature,
    predictor=predictor, free_raw_data=free_raw_data,
    info=info, used_indices=NULL), class="lgb.Dataset")
  return(ret)
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
create.valid.lgb.Dataset <- function(dataset, data, info=list(),  ...) {
  info <- append(info, list(...))
  ret <- lgb.Dataset(data, dataset$params,
    dataset, dataset$categorical_feature, dataset$predictor,
    dataset$free_raw_data, info)
  return(ret)
}

# internal utility function
lgb.Dataset.get.handle <- function(dataset) {
  if(!is.null(dataset$handle)){
    return(dataset$handle)
  }
  # lazy load for handle
  cnames <- NULL
  if (is.matrix(dataset$raw_data) | class(dataset$raw_data) == "dgCMatrix") {
    cnames <- colnames(dataset$raw_data)
  } else {
    stop(paste("lgb.Dataset.get.handle: does not support to construct from ", typeof(dataset$raw_data)))
  }

  if(!is.null(dataset$categorical_feature)){
    fname_dict <- list()
    if(!is.null(cnames)){
      fname_dict <- as.list(setNames(0:(length(cnames)-1), cnames))
    }
    cate_indices <- list()
    for(key in dataset$categorical_feature){
      if(is.character(key)){
        idx <- fname_dict[[key]]
        if(is.null(idx)){
          stop(paste("lgb.Dataset.get.handle: cannot find feature name ", key))
        }
        cate_indices <- append(cate_indices, idx)
      } else {
        # one-based indices to zero-based
        idx <- as.integer(key - 1)
        cate_indices <- append(cate_indices, idx)
      }
    }
    dataset$categorical_feature <- cate_indices
  }
  has_header <- FALSE
  if (!is.null(dataset$params$has_header) | !is.null(dataset$params$header)) {
    if (tolower(as.character(dataset$params$has_header)) == "true" | tolower(as.character(dataset$params$header)) == "true") {
      has_header <- TRUE
    }
  }
  params_str <- as.character(lgb.params2str(dataset$params))
  ref_handle <- NULL

  if(!is.null(dataset$reference)){
    ref_handle <- lgb.Dataset.get.handle(dataset$reference)
  }
  if(is.null(dataset$used_indices)){
    if (typeof(dataset$raw_data) == "character") {
      handle <- .Call("LGBM_DatasetCreateFromFile_R", dataset$raw_data, params_str, ref_handle,
        PACKAGE = "lightgbm")
    } else if (is.matrix(dataset$raw_data)) {
      handle <- .Call("LGBM_DatasetCreateFromMat_R", dataset$raw_data, params_str, ref_handle,
        PACKAGE = "lightgbm")
    } else if (class(dataset$raw_data) == "dgCMatrix") {
      handle <- .Call("LGBM_DatasetCreateFromCSC_R", dataset$raw_data@p, 
        dataset$raw_data@i, dataset$raw_data@x, nrow(dataset$raw_data),
        params_str, ref_handle, PACKAGE = "lightgbm")
    } else {
      stop(paste("lgb.Dataset.get.handle: does not support to construct from ",
                 typeof(dataset$raw_data)))
    }
  } else {
    if(is.null(dataset$reference)) {
      stop("lgb.Dataset.get.handle: refenence cannot be NULL")
    }
    handle <- .Call("LGBM_DatasetGetSubset_R", ref_handle, dataset$used_indices, 
      params_str, PACKAGE = "lightgbm")
  }

  class(handle) <- "lgb.Dataset.handle"
  dataset$handle <- handle
  # set feature names
  if(!is.null(cnames)){
    dataset$colnames <- as.list(cnames)
    .Call("LGBM_DatasetSetFeatureNames_R", dataset$handle, dataset$colnames, PACKAGE="lightgbm")
  }
  if(!is.null(dataset$predictor) & is.null(dataset$used_indices)){
    # load init score
    init_score <- lgb.Predictor.predict(dataset$predictor, dataset$raw_data, rawscore=TRUE, reshape=TRUE)
    # not need to transpose, for is col_marjor
    init_score <- as.vector(init_score)
    dataset$info$init_score <- init_score
  }

  if(dataset$free_raw_data){
    dataset$raw_data <- NULL
  }

  if (length(dataset$info) > 0){
    # set infos
    for (i in 1:length(dataset$info)) {
      p <- dataset$info[i]
      setinfo(dataset, names(p), p[[1]])
    }
  }

  if(is.null(getinfo(dataset,"label"))){
    stop("label should be set")
  }

  return(dataset$handle)
}

#' Construct Dataset explicit
#' 
#' Returns NULL
#' @param dataset Object of class \code{lgb.Dataset}
#' 
#' @export
construct.lgb.Dataset <- function(dataset) {
  lgb.Dataset.get.handle(dataset)
}

#' Dimensions of lgb.Dataset
#' 
#' Returns a vector of numbers of rows and of columns in an \code{lgb.Dataset}.
#' @param x Object of class \code{lgb.Dataset}
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
dim.lgb.Dataset <- function(x) {
  if(!is.null(x$handle)) {
    return(c(.Call("LGBM_DatasetGetNumData_R", x$handle, PACKAGE="lightgbm"),
    .Call("LGBM_DatasetGetNumFeature_R", x$handle, PACKAGE="lightgbm")))
  } else if (is.matrix(dataset$raw_data) | class(dataset$raw_data) == "dgCMatrix") {
    return(dim(x$raw_data))
  } else {
    stop("cannot get Dimensions before dataset constructed, please call construct.lgb.Dataset explicit")
  }
}

#' Handling of column names of \code{lgb.Dataset}
#' @param x object of class \code{lgb.Dataset}
#' @rdname colnames.lgb.Dataset
#' @export
colnames.lgb.Dataset <- function(x) {
  return(x$colnames)
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
slice <- function(object, ...) UseMethod("slice")

#' @rdname slice.lgb.Dataset
#' @export
slice.lgb.Dataset <- function(object, idxset, ...) {
  if (class(object) != "lgb.Dataset") {
    stop("slice.lgb.Dataset: first argument dtrain must be lgb.Dataset")
  }
  ret <- structure(list(handle=NULL, raw_data=NULL,
    params=object$params, reference=object, 
    categorical_feature=object$categorical_feature,
    predictor=object$predictor, free_raw_data=object$free_raw_data,
    info=NULL, used_indices=idxset), class="lgb.Dataset")
  return(ret)
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
getinfo <- function(object, ...) UseMethod("getinfo")

#' @rdname getinfo
#' @export
getinfo.lgb.Dataset <- function(object, name, ...) {
  if (typeof(name) != "character" ||
      length(name) != 1 ||
      !name %in% c('label', 'weight', 'init_score', 'group')) {
    stop("getinfo: name must one of the following\n",
         "    'label', 'weight', 'init_score', 'group'")
  }
  if(is.null(object$info$name)){
    ret <- .Call("LGBM_DatasetGetField_R", object$handle, name, PACKAGE = "lightgbm")
    if (length(ret) > 0) {
      object$info$name <- ret
    }
  }
  return(object$info$name)
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
setinfo <- function(object, ...) UseMethod("setinfo")

#' @rdname setinfo
#' @export
setinfo.lgb.Dataset <- function(object, name, info, ...) {
  if (typeof(name) != "character" ||
      length(name) != 1 ||
      !name %in% c('label', 'weight', 'init_score', 'group')) {
    stop("setinfo: name must one of the following\n",
         "    'label', 'weight', 'init_score', 'group'")
    return(FALSE)
  }
  if (name == "group") {
    info <- as.integer(info)
  } else {
    info <- as.numeric(info)
  }
  object$info$name <- info
  if (!is.null(object$handle)) {
    .Call("LGBM_DatasetSetField_R", object$handle, name, object$info$name,
      PACKAGE = "lightgbm")
  }
  return(TRUE)
}

#' set categorical feature of \code{lgb.Dataset}
#' @param x object of class \code{lgb.Dataset}
#' @param categorical_feature categorical features
#' @rdname set.categorical.feature.lgb.Dataset
#' @export
set.categorical.feature.lgb.Dataset <- function(x, categorical_feature) {
  if(is.null(x$raw_data)){
    stop("cannot set categorical feature after free raw data,
         please set free_raw_data=FALSE when construct lgb.Dataset")
  }
  x$categorical_feature = categorical_feature
  x$handle <- NULL
}

# internal utility function
lgb.Dataset.set.predictor <- function(x, predictor) {
  if(is.null(x$raw_data)){
    stop("cannot set predictor after free raw data,
         please set free_raw_data=FALSE when construct lgb.Dataset")
  }
  if(!typeof(predictor) == "lgb.Predictor"){
     stop("Only can use lgb.Predictor as predictor")
  }
  x$predictor = predictor
  x$handle <- NULL
}

#' set reference of \code{lgb.Dataset}
#' @param x object of class \code{lgb.Dataset}
#' @param reference object of class \code{lgb.Dataset}
#' @rdname set.reference.lgb.Dataset
#' @export
set.reference.lgb.Dataset <- function(x, reference) {
  if(is.null(x$raw_data)){
    stop("cannot set reference after free raw data,
         please set free_raw_data=FALSE when construct lgb.Dataset")
  }
  if(!typeof(reference) == "lgb.Dataset"){
    stop("Only can use lgb.Dataset as reference")
  }
  x$reference = reference
  x$handle <- NULL
}

#' save \code{lgb.Dataset} to binary file
#' @param x object of class \code{lgb.Dataset}
#' @param fname object filename of output file
#' @rdname save.binary.lgb.Dataset
#' @export
save.binary.lgb.Dataset <- function(x, fname) {
  .Call("LGBM_DatasetSaveBinary_R", x$handle, fname, PACKAGE = "lightgbm")
}
