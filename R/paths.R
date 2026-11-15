# Data path resolution (R side); serves the same purpose as each Python
# section's paths.py.
#
# Usage (each script's header block sources this file automatically):
#     data_file("data_ml.RData")
#     load(data_file("data_ml.RData"))


#' Walk up from a given starting point to find the repo root (the first
#' ancestor that contains both a data/ and a tools/ directory)
find_repo_root <- function(start = getwd()) {
  d <- normalizePath(start, mustWork = FALSE)
  repeat {
    if (dir.exists(file.path(d, "data")) && dir.exists(file.path(d, "tools"))) {
      return(d)
    }
    parent <- dirname(d)
    if (identical(parent, d)) {
      stop("Could not locate the repo root: please run this script from inside the repo.", call. = FALSE)
    }
    d <- parent
  }
}

# Each script's header block resolves the repo root from the script's own
# location (independent of the working directory) and stores it in
# .qml_repo_root before sourcing this file. If this file is sourced
# directly without going through a header block, fall back to searching
# upward from the working directory.
REPO_ROOT <- if (exists(".qml_repo_root", envir = globalenv())) {
  get(".qml_repo_root", envir = globalenv())
} else {
  find_repo_root()
}

#' The single location where all of the book's data lives
DATA_DIR <- file.path(REPO_ROOT, "data")

#' Return the full path to a data file under data/, with an actionable
#' error message if it's missing
data_file <- function(name) {
  path <- file.path(DATA_DIR, name)
  if (!file.exists(path)) {
    stop(sprintf(
      "Data file not found: %s\nPlease set up the data/ directory as described in the top-level README.md's 'Data' section.",
      path
    ), call. = FALSE)
  }
  path
}
