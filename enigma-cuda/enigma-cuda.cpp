#include "runner.h"

int main(int argc, char *argv[])
{
    Runner runner;
    bool ok;

    ok = runner.settings.FromCommandLine(argc, argv);
    if (!ok) return 1;

    ok = runner.Initialize();
    if (!ok) return 2;

    ok = runner.Run();
    if (!ok) return 3;

    return 0;
}