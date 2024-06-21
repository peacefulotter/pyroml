from .utils import get_lr


# TODO: move these methods to a Console Logger class
class Statistics:
    def log_stats(self, stats, epoch, iteration):
        log = f"[epoch] {epoch:03d} | [iter] {iteration:05d}:{self.config.max_iterations:05d}"
        for metric in self.train_metrics:
            log += f" | [{metric.name}] tr: {stats['train'][metric.name]:.4f}"
            if "eval" in stats:
                log += f", ev: {stats['eval'][metric.name]:.4f}"
        log += f" | [lr] {stats['lr']:.4f}"
        self.logger.log(log)

    def register(self, train_output, train_target, train_loss, epoch, iteration):

        self.lr = get_lr(self.config, self.scheduler)
        stats = {"epoch": epoch, "iter": iteration, "train": train_stats, "lr": self.lr}

        if self.config.verbose:
            self.log_stats(stats, epoch, iteration)

        return stats
