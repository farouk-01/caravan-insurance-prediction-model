library(Gifi)
ex_catpca <- function(
  df
) {
  data_pca <- df[, c("x_ord", "x_ord2", "z", "z_corr")]
  levels <- c("ordinal", "ordinal", "metric", "metric")

  data_pca$z <- scale(data_pca$z)
  data_pca$z_corr <- scale(data_pca$z_corr)

  res_catpca <- princals(data_pca, levels = levels)
  z_cat <- res_catpca$objectscores

  mask_tp <- df$y_ord > 3
  cols <- ifelse(mask_tp, "steelblue", "tomato")

  plot(z_cat[, 1], z_cat[, 2],
    col = adjustcolor(cols, alpha.f = 0.5),
    pch = 16,
    cex = 1.5,
    main = "CATPCA variable ordinale",
    xlab = "CATPC1",
    ylab = "CATPC2"
  )

  legend("topright",
    legend = c("TP", "FN"),
    col = c("steelblue", "tomato"), pch = 16, bty = "n"
  )
}
