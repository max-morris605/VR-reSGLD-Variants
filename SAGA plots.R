effectiveSize(SAGA.VR.reSGLD_chain)

plot(ts(SAGA.VR.reSGLD_chain))

library(ggplot2)

# KDE of your chain
p <- ggplot(SAGA.VR.reSGLD_chain, aes(x = V1)) +
  geom_density(aes(color = "Chain KDE"), linewidth = 1)

xgrid <- data.frame(
  x = seq(min(SAGA.VR.reSGLD_chain$V1),
          max(SAGA.VR.reSGLD_chain$V1), length.out = 1000)
)

p + geom_line(
      data = xgrid,
      aes(x = x,
          y = 0.5 * dnorm(x, -5, sqrt(5)) + 0.5 * dnorm(x, 25, sqrt(5)),
          color = "True Mixture"),
      linewidth = 1
    ) +
  scale_color_manual(
    name = "Distribution",
    values = c("Chain KDE" = "black", "True Mixture" = "red")
  ) +
  labs(title = "Chain KDE with True Mixture Density",
       x = "beta", y = "Density") +
  theme_minimal()


acf(SAGA.VR.reSGLD_chain$V1)
