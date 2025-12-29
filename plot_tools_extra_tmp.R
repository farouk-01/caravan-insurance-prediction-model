library(Gifi)

plot_catpca <- function(
  df, var1, var2,
  ndim = 2, num_cols = NULL, group_col = "Group"
) {
  df_x <- df[df[[group_col]] %in% c(var1, var2), , drop = FALSE]
  grp <- df_x[[group_col]]

  is_non_unique <- sapply(df_x, function(x) length(unique(x[!is.na(x)])) < 2)
  non_unique_cols <- names(is_non_unique)[is_non_unique]
  cols_to_keep <- setdiff(colnames(df_x), c(group_col, non_unique_cols))

  df_x <- df_x[, cols_to_keep, drop = FALSE]

  level <- rep("ordinal", ncol(df_x))
  names(level) <- colnames(df_x)

  if (!is.null(num_cols)) level[num_cols] <- "metric"

  fit_ord <- princals(df_x, ndim = ndim, levels = level)
  cols <- ifelse(grp == var1, "steelblue", "tomato")

  plot(fit_ord$objectscores[, 1], fit_ord$objectscores[, 2],
    pch = 16, cex = 1.5, col = adjustcolor(cols, alpha.f = 0.5),
    xlab = "CATPC1", ylab = "CATPC2", asp = 1, main = "CATPCA Plot"
  )

  points(aggregate(fit_ord$objectscores, list(grp), mean)[, 2:3],
    pch = 8, cex = 2, lwd = 2
  )
  legend(
    "topright",
    legend = c(var1, var2),
    col = c("steelblue", "tomato"),
    pch = 16, bty = "n"
  )

  invisible(list(
    scores = fit_ord$objectscores,
    loadings = fit_ord$loadings
  ))
}
