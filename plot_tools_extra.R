library(Gifi)

plot_catpca <- function(
  df, cols, var1, var2,
  ndim = 2, num_cols = NULL, group_col = "Group"
) {
  df_sub <- df[df[[group_col]] %in% c(var1, var2), ]

  df_x <- df_sub[, cols]
  grp <- df_sub[[group_col]]

  level <- rep("ordinal", ncol(df_x))
  names(level) <- colnames(df_x)

  if (!is.null(num_cols)) {
    level[num_cols] <- "numeric"
  }

  fit_ord <- princals(df_x, ndim = ndim, level = level)
  plot(fit_ord, plot.type = "biplot", main = "Graph CATPCA")

  invisible(list(
    scores = fit_ord$objectscores,
    loadings = fit_ord$loadings,
    groups = grp
  ))
}
