/* Mach-O entry point for Audio Separator.app.

A shell script as CFBundleExecutable is not a Mac app: Gatekeeper calls the
bundle damaged, and the Python GUI is not the process Launch Services started.
This binary is that process. It prepares folders and env, then replaces itself
with the embedded python running desktop.py.
*/
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <mach-o/dyld.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static void mkdir_p(const char *path) {
    char tmp[PATH_MAX];
    size_t len;
    size_t i;

    if (snprintf(tmp, sizeof tmp, "%s", path) >= (int)sizeof tmp) {
        return;
    }
    len = strlen(tmp);
    for (i = 1; i < len; i++) {
        if (tmp[i] != '/') {
            continue;
        }
        tmp[i] = '\0';
        if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
            return;
        }
        tmp[i] = '/';
    }
    if (mkdir(tmp, 0755) != 0 && errno != EEXIST) {
        return;
    }
}

static int parent_dir(char *path) {
    char *slash = strrchr(path, '/');
    if (slash == NULL || slash == path) {
        return -1;
    }
    *slash = '\0';
    return 0;
}

static void tell_user_failed(void) {
    system(
        "/usr/bin/osascript -e 'display dialog "
        "\"Audio Separator no pudo abrir. El detalle está en "
        "~/Library/Logs/Audio Separator/launch.log\" "
        "buttons {\"OK\"} default button 1 with title \"Audio Separator\"'"
    );
}

static void write_models_note(const char *data) {
    char path[PATH_MAX];
    FILE *handle;

    if (snprintf(path, sizeof path, "%s/DONDE-VAN-LOS-MODELOS.txt", data) >= (int)sizeof path) {
        return;
    }
    if (access(path, F_OK) == 0) {
        return;
    }
    handle = fopen(path, "w");
    if (handle == NULL) {
        return;
    }
    fprintf(handle, "Los modelos grandes no vienen con Audio Separator. La app no los descarga.\n\n");
    fprintf(handle, "Separación local: funciona sin archivos extra (canal central).\n\n");
    fprintf(handle, "Separación mejor, un archivo .onnx:\n%s/mdx_models\n\n", data);
    fprintf(handle, "Voz RVC:\n");
    fprintf(handle, "%s/rvc_models/hubert_base/   (config.json y los pesos)\n", data);
    fprintf(handle, "%s/rvc_models/rmvpe.pt\n", data);
    fprintf(handle, "%s/rvc_models/tu-voz.pth\n", data);
    fprintf(handle, "%s/rvc_models/tu-voz.index   (opcional)\n\n", data);
    fprintf(handle, "No hace falta una GPU NVIDIA. La separación corre en CPU.\n");
    fprintf(handle, "La conversión de voz usa el chip de Apple si PyTorch lo detecta; si no, CPU.\n");
    fclose(handle);
}

static int open_log(const char *log_dir) {
    char path[PATH_MAX];
    int fd;
    time_t now;

    mkdir_p(log_dir);
    if (snprintf(path, sizeof path, "%s/launch.log", log_dir) >= (int)sizeof path) {
        return -1;
    }
    fd = open(path, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) {
        return -1;
    }
    now = time(NULL);
    dprintf(fd, "\n---- %s", ctime(&now));
    return fd;
}

int main(void) {
    char exe[PATH_MAX];
    char contents[PATH_MAX];
    char resources[PATH_MAX];
    char py[PATH_MAX];
    char script[PATH_MAX];
    char bin_dir[PATH_MAX];
    char data[PATH_MAX];
    char log_dir[PATH_MAX];
    char path_env[PATH_MAX * 2];
    char child[PATH_MAX];
    const char *home;
    const char *old_path;
    uint32_t size = sizeof exe;
    int log_fd;

    if (_NSGetExecutablePath(exe, &size) != 0 || realpath(exe, contents) == NULL) {
        tell_user_failed();
        return 1;
    }
    /* .../Contents/MacOS/audio-separator -> .../Contents */
    if (parent_dir(contents) != 0 || parent_dir(contents) != 0) {
        tell_user_failed();
        return 1;
    }
    if (snprintf(resources, sizeof resources, "%s/Resources", contents) >= (int)sizeof resources) {
        tell_user_failed();
        return 1;
    }
    if (snprintf(py, sizeof py, "%s/python/bin/python3", resources) >= (int)sizeof py ||
        snprintf(script, sizeof script, "%s/app/desktop.py", resources) >= (int)sizeof script ||
        snprintf(bin_dir, sizeof bin_dir, "%s/bin", resources) >= (int)sizeof bin_dir) {
        tell_user_failed();
        return 1;
    }

    home = getenv("HOME");
    if (home == NULL || home[0] == '\0') {
        home = "/tmp";
    }
    if (snprintf(data, sizeof data, "%s/Library/Application Support/Audio Separator", home) >= (int)sizeof data ||
        snprintf(log_dir, sizeof log_dir, "%s/Library/Logs/Audio Separator", home) >= (int)sizeof log_dir) {
        tell_user_failed();
        return 1;
    }

    mkdir_p(data);
    snprintf(child, sizeof child, "%s/mdx_models", data);
    mkdir_p(child);
    snprintf(child, sizeof child, "%s/rvc_models", data);
    mkdir_p(child);
    snprintf(child, sizeof child, "%s/downloads", data);
    mkdir_p(child);
    snprintf(child, sizeof child, "%s/clean_song_output", data);
    mkdir_p(child);
    snprintf(child, sizeof child, "%s/remix_output", data);
    mkdir_p(child);
    snprintf(child, sizeof child, "%s/rvc_output", data);
    mkdir_p(child);
    write_models_note(data);

    if (access(py, X_OK) != 0 || access(script, R_OK) != 0) {
        tell_user_failed();
        return 1;
    }

    setenv("AUDIO_SEPARATOR_HOME", data, 1);
    setenv("PYTHONNOUSERSITE", "1", 1);
    setenv("PYTHONDONTWRITEBYTECODE", "1", 1);
    setenv("PYTHONIOENCODING", "utf-8", 1);
    old_path = getenv("PATH");
    if (old_path == NULL) {
        old_path = "/usr/bin:/bin";
    }
    snprintf(path_env, sizeof path_env, "%s:/opt/homebrew/bin:/usr/local/bin:%s", bin_dir, old_path);
    setenv("PATH", path_env, 1);

    log_fd = open_log(log_dir);
    if (log_fd >= 0) {
        dup2(log_fd, STDOUT_FILENO);
        dup2(log_fd, STDERR_FILENO);
        if (log_fd > STDERR_FILENO) {
            close(log_fd);
        }
    }

    (void)chdir(data);
    execl(py, "Audio Separator", "-u", script, (char *)NULL);
    dprintf(STDERR_FILENO, "No pude ejecutar %s: %s\n", py, strerror(errno));
    tell_user_failed();
    return 1;
}
