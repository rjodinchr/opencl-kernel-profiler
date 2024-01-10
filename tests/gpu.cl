void kernel inc(global int *buffer)
{
    size_t gid = get_global_id(0);
    buffer[gid]++;
}
