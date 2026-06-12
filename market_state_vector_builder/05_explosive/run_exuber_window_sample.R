args <- commandArgs(trailingOnly = TRUE)

get_arg <- function(name, default = NULL) {
  key <- paste0("--", name)
  pos <- match(key, args)
  if (is.na(pos) || pos == length(args)) {
    return(default)
  }
  args[[pos + 1]]
}

input_path <- get_arg("input", "market_state_vector_builder/outputs/shape_state_analysis_manual/exuber_sample/input/exuber_windows.csv")
out_dir <- get_arg("out-dir", "market_state_vector_builder/outputs/shape_state_analysis_manual/exuber_sample/raw_r_results")
minw <- as.integer(get_arg("minw", "50"))
lag <- as.integer(get_arg("lag", "1"))
nrep <- as.integer(get_arg("nrep", "499"))
seed <- as.integer(get_arg("seed", "42"))
max_windows_arg <- get_arg("max-windows", "")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
suppressPackageStartupMessages(library(exuber))

df <- read.csv(input_path, stringsAsFactors = FALSE)
window_ids <- unique(df$window_id)
if (nzchar(max_windows_arg)) {
  window_ids <- head(window_ids, as.integer(max_windows_arg))
}

cv_cache <- new.env(parent = emptyenv())
records <- list()

for (i in seq_along(window_ids)) {
  wid <- window_ids[[i]]
  g <- df[df$window_id == wid, ]
  g <- g[order(g$t), ]
  x <- g$log_close
  n <- length(x)
  if (n <= minw + lag + 5 || sd(x) <= 1e-12) {
    records[[length(records) + 1]] <- data.frame(
      window_id = wid,
      n_obs = n,
      minw = minw,
      lag = lag,
      nrep = nrep,
      gsadf_stat = NA_real_,
      gsadf_cv95 = NA_real_,
      gsadf_reject = FALSE,
      error = "too_short_or_constant"
    )
    next
  }

  cache_key <- paste(n, minw, sep = "_")
  if (!exists(cache_key, envir = cv_cache, inherits = FALSE)) {
    assign(cache_key, radf_mc_cv(n = n, minw = minw, nrep = nrep, seed = seed), envir = cv_cache)
  }
  cv <- get(cache_key, envir = cv_cache, inherits = FALSE)

  result <- tryCatch({
    fit <- radf(x, minw = minw, lag = lag)
    gsadf_stat <- as.numeric(fit$gsadf[[1]])
    gsadf_cv95 <- as.numeric(cv$gsadf_cv[["95%"]])
    data.frame(
      window_id = wid,
      n_obs = n,
      minw = minw,
      lag = lag,
      nrep = nrep,
      gsadf_stat = gsadf_stat,
      gsadf_cv95 = gsadf_cv95,
      gsadf_reject = is.finite(gsadf_stat) && gsadf_stat > gsadf_cv95,
      error = ""
    )
  }, error = function(e) {
    data.frame(
      window_id = wid,
      n_obs = n,
      minw = minw,
      lag = lag,
      nrep = nrep,
      gsadf_stat = NA_real_,
      gsadf_cv95 = NA_real_,
      gsadf_reject = FALSE,
      error = conditionMessage(e)
    )
  })

  records[[length(records) + 1]] <- result
  if (i %% 100 == 0) {
    cat("finished", i, "of", length(window_ids), "\n")
  }
}

out <- do.call(rbind, records)
write.csv(out, file.path(out_dir, "exuber_window_results.csv"), row.names = FALSE)
cat("Windows:", nrow(out), "\n")
cat("Reject rate:", mean(out$gsadf_reject, na.rm = TRUE), "\n")
cat("Outputs written to:", out_dir, "\n")
