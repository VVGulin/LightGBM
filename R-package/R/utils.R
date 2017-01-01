
lgb.params2str <- function(params, ...) {
  if (typeof(params) != "list") 
    stop("params must be a list")
  names(params) <- gsub("\\.", "_", names(params))
  # merge parameters from the params and the dots-expansion
  dot_params <- list(...)
  names(dot_params) <- gsub("\\.", "_", names(dot_params))
  if (length(intersect(names(params),
                       names(dot_params))) > 0)
    stop("Same parameters in 'params' and in the call are not allowed. Please check your 'params' list.")
  params <- c(params, dot_params)
  ret <- list()
  ret <- c(ret, "")
  for( key in names(params) ) {
    # join multi value first
    val <- paste0(params[[key]], collapse=",")
    # join key value
    pair <- paste0(c(key, val), collapse="=")
    ret <- c(ret, pair)
  }
  
  return(paste0(ret, collapse=" "))
}

lgb.check.r6.class <- function(object, name) {
  if(!("R6" %in% class(object))){
    return(FALSE)
  }
  if(!(name %in% class(object))){
    return(FALSE)
  }
  return(TRUE)
}