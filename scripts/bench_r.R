# Bioconductor S4Vectors/IRanges timings for the workload in
# scripts/bench.py and crates/runlength/examples/bench.rs.
#
#     Rscript scripts/bench_r.R
#
# Each timing is the best of three runs.
suppressPackageStartupMessages({
  library(S4Vectors)
  library(IRanges)
  library(GenomicRanges)
})
set.seed(42)
N <- 1e6
# R recycles the shorter Rle in arithmetic, so give both the same length.
WIDTH <- 1e8 + 100

timed <- function(label, f) {
  best <- Inf
  for (i in 1:3) {
    t <- proc.time()[["elapsed"]]
    out <- f()
    best <- min(best, proc.time()[["elapsed"]] - t)
  }
  cat(sprintf("%-40s %8.0f ms\n", label, best * 1e3))
  invisible(out)
}

starts <- sample.int(1e8, N, replace = TRUE)
starts2 <- sample.int(1e8, N, replace = TRUE)
slice_starts <- sample.int(99e6, N, replace = TRUE)
seqs <- paste0("k", sample.int(25, N, replace = TRUE))

a <- timed("coverage, 1M intervals",
           function() coverage(IRanges(starts, width = 100), width = WIDTH))
b <- coverage(IRanges(starts2, width = 100), width = WIDTH)
cat(sprintf("  runs: %d and %d\n", nrun(a), nrun(b)))
s <- timed("a + b", function() a + b)
cat(sprintf("  runs: %d\n", nrun(s)))
timed("a < b", function() a < b)
timed("a.mean()", function() mean(a))
views <- Views(a, IRanges(slice_starts, width = 1000))
x <- timed("1M slices of 1000, copied", function() as(views, "RleList"))
cat(sprintf("  total length: %.0f\n", sum(as.numeric(lengths(x)))))
timed("sum of 1M slices of 1000", function() viewSums(views))
gr <- GRanges(seqs, IRanges(starts, width = 100))
batch <- timed("coverage, 1M intervals, 25 keys", function() coverage(gr))
timed("batch + batch, 25 keys", function() batch + batch)
