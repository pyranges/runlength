# Bioconductor S4Vectors/IRanges timings for the workload in
# crates/runlength/examples/bench.rs.
#
#     Rscript scripts/bench_r.R
suppressPackageStartupMessages({
  library(S4Vectors)
  library(IRanges)
  library(GenomicRanges)
})
set.seed(42)
N <- 1e6

timed <- function(label, expr) {
  t <- proc.time()[["elapsed"]]
  out <- force(expr)
  cat(sprintf("%-40s %8.1f ms\n", label, (proc.time()[["elapsed"]] - t) * 1e3))
  invisible(out)
}

starts <- sample.int(1e8, N, replace = TRUE)
a <- timed("coverage_count, 1M intervals", coverage(IRanges(starts, width = 100)))
starts2 <- sample.int(1e8, N, replace = TRUE)
b <- timed("coverage_count, 1M more intervals", coverage(IRanges(starts2, width = 100)))
cat(sprintf("  runs: %d and %d\n", nrun(a), nrun(b)))
cat(sprintf("  memory: %.1f MB (%.1f bytes per run)\n",
            object.size(a) / 1e6, as.numeric(object.size(a)) / nrun(a)))

s <- timed("a + b", a + b)
cat(sprintf("  runs: %d\n", nrun(s)))
timed("a < b", a < b)
timed("a.mean()", mean(a))
af <- a * 1.0  # double values, as in pyrle and the f64 benchmarks
timed("a + b (double)", af + b)

slice_starts <- sample.int(99e6, N, replace = TRUE)
timed("getitems, 1M slices of 1000",
      as(Views(a, IRanges(slice_starts, width = 1000)), "RleList"))
timed("  viewSums over the same slices",
      viewSums(Views(a, IRanges(slice_starts, width = 1000))))

seqs <- paste0("k", sample.int(25, N, replace = TRUE))
gr <- GRanges(seqs, IRanges(starts, width = 100))
batch <- timed("coverage_sum_by_key, 25 keys", coverage(gr))
timed("batch + batch, 25 keys", batch + batch)
